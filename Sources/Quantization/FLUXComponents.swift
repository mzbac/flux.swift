import Foundation
import MLX
import MLXNN
import Logging

private let logger = Logger(label: "flux.swift.quantization.FLUXComponents")

public protocol FLUXComponents {
    var transformer: MultiModalDiffusionTransformer { get }
    var vae: VAE { get }
    var clipEncoder: CLIPEncoder { get }
    var t5Encoder: T5Encoder { get }
}

public struct ComponentInfo {
    let name: String
    let prefix: String
    let module: Module
}

public extension FLUXComponents {
    var components: [ComponentInfo] {
        return [
            ComponentInfo(name: "transformer", prefix: "transformer", module: transformer),
            ComponentInfo(name: "vae", prefix: "vae", module: vae),
            ComponentInfo(name: "text_encoder", prefix: "text_encoder", module: clipEncoder),
            ComponentInfo(name: "text_encoder_2", prefix: "text_encoder_2", module: t5Encoder)
        ]
    }
    
    func component(named name: String) -> Module? {
        switch name {
        case "transformer":
            return transformer
        case "vae":
            return vae
        case "clipEncoder", "text_encoder":
            return clipEncoder
        case "t5Encoder", "text_encoder_2":
            return t5Encoder
        default:
            return nil
        }
    }
    
    var hasQuantizedLayers: Bool {
        return components.contains { componentInfo in
            QuantizationUtils.hasQuantizedLayers(in: componentInfo.module)
        }
    }
    
    var quantizedLayerCount: Int {
        return components.reduce(0) { total, componentInfo in
            total + QuantizationUtils.countQuantizedLayers(in: componentInfo.module)
        }
    }
    
    func quantizeAllComponents(bits: Int = QuantizationUtils.defaultBits, 
                               groupSize: Int = QuantizationUtils.defaultGroupSize) {
        logger.info("Quantizing all components with \(bits)-bit, group size: \(groupSize)")
        
        quantize(model: transformer, groupSize: groupSize, bits: bits) { _, m in
            m is Linear && (m as? Linear)?.weight.shape[1] ?? 0 > 64
        }
        
        quantize(model: vae, groupSize: groupSize, bits: bits) { _, m in m is Linear }
        quantize(model: clipEncoder, groupSize: groupSize, bits: bits) { _, m in m is Linear }
        quantize(model: t5Encoder, groupSize: groupSize, bits: bits) { _, m in m is Linear }
        
        logger.info("Quantization complete", metadata: [
            "totalQuantizedLayers": .string(String(quantizedLayerCount)),
            "bits": .string(String(bits)),
            "groupSize": .string(String(groupSize))
        ])
    }
    
    func updateComponentWeights(from weights: [String: MLXArray]) {
        let componentWeights = ComponentWeights.split(weights)
        
        if !componentWeights.vae.isEmpty {
            logger.debug("Updating VAE weights", metadata: ["parameterCount": .string(String(componentWeights.vae.count))])
            vae.update(parameters: ModuleParameters.unflattened(componentWeights.vae))
        }
        
        if !componentWeights.transformer.isEmpty {
            logger.debug("Updating transformer weights", metadata: ["parameterCount": .string(String(componentWeights.transformer.count))])
            
            var filteredTransformerWeights = componentWeights.transformer
            let modelType = String(describing: type(of: self))
            let hasGuidanceEmbedder = modelType.contains("Dev") || modelType.contains("Kontext")
            
            let guidanceKeys = filteredTransformerWeights.keys.filter { $0.contains("guidance_embedder") }
            if !guidanceKeys.isEmpty && !hasGuidanceEmbedder {
                logger.debug("Removing guidance_embedder weights", metadata: ["count": .string(String(guidanceKeys.count)), "modelType": .string(modelType)])
                guidanceKeys.forEach { filteredTransformerWeights.removeValue(forKey: $0) }
            }
            
            transformer.update(parameters: ModuleParameters.unflattened(filteredTransformerWeights))
        }
        
        if !componentWeights.clipEncoder.isEmpty {
            logger.debug("Updating CLIP encoder weights", metadata: ["parameterCount": .string(String(componentWeights.clipEncoder.count))])
            clipEncoder.update(parameters: ModuleParameters.unflattened(componentWeights.clipEncoder))
        }
        
        if !componentWeights.t5Encoder.isEmpty {
            logger.debug("Updating T5 encoder weights", metadata: ["parameterCount": .string(String(componentWeights.t5Encoder.count))])
            t5Encoder.update(parameters: ModuleParameters.unflattened(componentWeights.t5Encoder))
        }
    }
    
    func applySelectiveQuantization(weights: [String: MLXArray], bits: Int, groupSize: Int) {
        logger.info("Applying selective quantization")
        
        for componentInfo in components {
            logger.debug("Checking component", metadata: ["component": .string(componentInfo.name)])
            
            var quantizedCount = 0
            for key in weights.keys {
                if key.hasPrefix("\(componentInfo.prefix).") && key.hasSuffix(".scales") {
                    quantizedCount += 1
                }
            }
            
            if quantizedCount > 0 {
                logger.debug("Found pre-quantized layers", metadata: ["count": .string(String(quantizedCount)), "component": .string(componentInfo.name)])
                
                quantize(model: componentInfo.module) { path, module in
                    let fullPath = "\(componentInfo.prefix).\(path)"
                    if weights["\(fullPath).scales"] != nil {
                        return (groupSize: groupSize, bits: bits)
                    } else {
                        return nil
                    }
                }
            } else {
                logger.debug("No pre-quantized weights found", metadata: ["component": .string(componentInfo.name)])
            }
        }
    }
}

struct ComponentWeights {
    let vae: [String: MLXArray]
    let transformer: [String: MLXArray]
    let clipEncoder: [String: MLXArray]
    let t5Encoder: [String: MLXArray]
    
    static func split(_ weights: [String: MLXArray]) -> ComponentWeights {
        var vae: [String: MLXArray] = [:]
        var transformer: [String: MLXArray] = [:]
        var clipEncoder: [String: MLXArray] = [:]
        var t5Encoder: [String: MLXArray] = [:]
        
        for (key, value) in weights {
            if key.hasPrefix("vae.") {
                vae[String(key.dropFirst(4))] = value
            } else if key.hasPrefix("transformer.") {
                transformer[String(key.dropFirst(12))] = value
            } else if key.hasPrefix("text_encoder.") {
                clipEncoder[String(key.dropFirst(13))] = value
            } else if key.hasPrefix("text_encoder_2.") {
                t5Encoder[String(key.dropFirst(15))] = value
            }
        }
        
        return ComponentWeights(
            vae: vae,
            transformer: transformer,
            clipEncoder: clipEncoder,
            t5Encoder: t5Encoder
        )
    }
}