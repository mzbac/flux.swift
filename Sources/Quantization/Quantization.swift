import Foundation
import MLX
import MLXNN
import Hub
import Logging

private let logger = Logger(label: "flux.swift.quantization.Quantization")

public enum Quantization {
    
    public static func save(
        model: FLUXComponents,
        to path: URL,
        bits: Int = QuantizationUtils.defaultBits,
        groupSize: Int = QuantizationUtils.defaultGroupSize
    ) throws {
        try QuantizationUtils.validateQuantizationParameters(bits: bits, groupSize: groupSize)
        
        if !model.hasQuantizedLayers {
            logger.info("Model is not quantized. Applying quantization...")
            model.quantizeAllComponents(bits: bits, groupSize: groupSize)
        }
        
        try QuantizationUtils.saveQuantizedModel(model, to: path, quantizationBits: bits, groupSize: groupSize)
    }
    
    @available(*, deprecated, message: "Use FLUX.loadQuantized() directly instead")
    public static func load(
        from source: String,
        modelType: String = "schnell",
        hub: HubApi = HubApi()
    ) async throws -> FLUX {
        return try await FLUX.loadQuantized(
            from: source,
            modelType: modelType,
            hub: hub
        )
    }
    
    public static func quantize(
        model: FLUXComponents,
        bits: Int = QuantizationUtils.defaultBits,
        groupSize: Int = QuantizationUtils.defaultGroupSize
    ) throws {
        try QuantizationUtils.validateQuantizationParameters(bits: bits, groupSize: groupSize)
        model.quantizeAllComponents(bits: bits, groupSize: groupSize)
    }
    
    public static func analyzeModel(_ model: FLUXComponents) -> QuantizationStats {
        var totalParams = 0
        var quantizedParams = 0
        var originalSize = 0
        var quantizedSize = 0
        
        for component in model.components {
            let params = QuantizationUtils.extractAllWeights(from: component.module)
            
            for (key, array) in params {
                let paramCount = array.size
                let paramSize = QuantizationUtils.sizeInBytes(array)
                
                totalParams += paramCount
                originalSize += paramCount * MemoryLayout<Float32>.size
                
                if QuantizationUtils.isQuantizedWeight(key: key) || array.dtype == .uint32 {
                    quantizedParams += paramCount
                    quantizedSize += paramSize
                } else {
                    quantizedSize += paramSize
                }
            }
        }
        
        return QuantizationStats(
            totalParameters: totalParams,
            quantizedParameters: quantizedParams,
            originalSize: originalSize,
            quantizedSize: quantizedSize
        )
    }
    
    public static func isQuantizedModel(at path: URL) throws -> Bool {
        let metadataPath = path.appendingPathComponent("metadata.json")
        
        guard FileManager.default.fileExists(atPath: metadataPath.path) else {
            return false
        }
        
        let data = try Data(contentsOf: metadataPath)
        let metadata = try JSONDecoder().decode(ModelMetadata.self, from: data)
        
        return metadata.quantizationBits > 0
    }
    
    public static func getModelMetadata(at path: URL) throws -> ModelMetadata {
        let metadataPath = path.appendingPathComponent("metadata.json")
        let data = try Data(contentsOf: metadataPath)
        return try JSONDecoder().decode(ModelMetadata.self, from: data)
    }
    
    public static func estimateQuantizedSize(
        model: FLUXComponents,
        bits: Int = QuantizationUtils.defaultBits
    ) -> Int {
        var estimatedSize = 0
        
        for component in model.components {
            let params = component.module.parameters().flattened()
            
            for (_, array) in params {
                if array.size > 1000 {
                    let quantizedSize = (array.size * bits) / 8
                    let scalesSize = (array.size / QuantizationUtils.defaultGroupSize) * MemoryLayout<Float32>.size
                    let biasesSize = scalesSize
                    estimatedSize += quantizedSize + scalesSize + biasesSize
                } else {
                    estimatedSize += QuantizationUtils.sizeInBytes(array)
                }
            }
        }
        
        return estimatedSize
    }
}

public extension FLUX {
    func saveQuantized(
        to path: URL,
        bits: Int = QuantizationUtils.defaultBits,
        groupSize: Int = QuantizationUtils.defaultGroupSize
    ) throws {
        guard let fluxModel = self as? FLUXComponents else {
            throw FluxError.modelComponentMissing
        }
        try Quantization.save(model: fluxModel, to: path, bits: bits, groupSize: groupSize)
    }
    
    var quantizationStats: QuantizationStats? {
        guard let fluxModel = self as? FLUXComponents else {
            return nil
        }
        return Quantization.analyzeModel(fluxModel)
    }
}