import Foundation
import Hub
import MLX
import MLXNN
import MLXRandom
import Tokenizers
import Logging

private let logger = Logger(label: "flux.swift.FluxModelCore")

public struct FluxModelConfiguration {
    public let transformerConfig: MultiModalDiffusionConfiguration
    public let t5Config: T5Configuration
    public let clipConfig: CLIPConfiguration
    public let vaeConfig: VAEConfiguration
    public let t5MaxSequenceLength: Int
    public let clipMaxSequenceLength: Int
    public let clipPaddingToken: Int32
    
    nonisolated(unsafe) public static let schnell = FluxModelConfiguration(
        transformerConfig: MultiModalDiffusionConfiguration(),
        t5Config: T5Configuration(),
        clipConfig: CLIPConfiguration(),
        vaeConfig: VAEConfiguration(),
        t5MaxSequenceLength: 256,
        clipMaxSequenceLength: 77,
        clipPaddingToken: 49407
    )
    
    nonisolated(unsafe) public static let dev = FluxModelConfiguration(
        transformerConfig: MultiModalDiffusionConfiguration(guidanceEmbeds: true),
        t5Config: T5Configuration(),
        clipConfig: CLIPConfiguration(),
        vaeConfig: VAEConfiguration(),
        t5MaxSequenceLength: 512,
        clipMaxSequenceLength: 77,
        clipPaddingToken: 49407
    )
    
    nonisolated(unsafe) public static let kontextDev = FluxModelConfiguration(
        transformerConfig: MultiModalDiffusionConfiguration(guidanceEmbeds: true),
        t5Config: T5Configuration(
            vocabSize: 32128,
            dModel: 4096,
            dKv: 64,
            dFf: 10240,
            numHeads: 64,
            numLayers: 24
        ),
        clipConfig: CLIPConfiguration(
            hiddenSize: 768,
            intermediateSize: 3072,
            headDimension: 64,
            batchSize: 1,
            numAttentionHeads: 12,
            positionEmbeddingsCount: 77,
            tokenEmbeddingsCount: 49408,
            numHiddenLayers: 11
        ),
        vaeConfig: VAEConfiguration(),
        t5MaxSequenceLength: 512,
        clipMaxSequenceLength: 77,
        clipPaddingToken: 49407
    )
}

public class FluxModelCore: @unchecked Sendable {
    public let transformer: MultiModalDiffusionTransformer
    public let vae: VAE
    public let t5Encoder: T5Encoder
    public let clipEncoder: CLIPEncoder
    
    var clipTokenizer: CLIPTokenizer
    var t5Tokenizer: any Tokenizer
    
    public let configuration: FluxModelConfiguration
    public var modelDirectory: URL?
    
    public init(hub: HubApi, fluxConfiguration: FluxConfiguration, modelConfiguration: FluxModelConfiguration) throws {
        self.configuration = modelConfiguration
        
        let repo = Hub.Repo(id: fluxConfiguration.id)
        let directory = hub.localRepoLocation(repo)
        
        (self.t5Tokenizer, self.clipTokenizer) = try FLUX.loadTokenizers(directory: directory, hub: hub)
        
        self.transformer = MultiModalDiffusionTransformer(modelConfiguration.transformerConfig)
        self.vae = VAE(modelConfiguration.vaeConfig)
        self.t5Encoder = T5Encoder(modelConfiguration.t5Config)
        self.clipEncoder = CLIPEncoder(modelConfiguration.clipConfig)
    }
    
    public init(hub: HubApi, modelDirectory: URL, modelConfiguration: FluxModelConfiguration) throws {
        self.configuration = modelConfiguration
        self.modelDirectory = modelDirectory
        
        logger.info("Initializing from quantized model directory: \(modelDirectory.path)")
        
        (self.t5Tokenizer, self.clipTokenizer) = try FLUX.loadTokenizers(directory: modelDirectory, hub: hub)
        
        self.transformer = MultiModalDiffusionTransformer(modelConfiguration.transformerConfig)
        self.vae = VAE(modelConfiguration.vaeConfig)
        self.t5Encoder = T5Encoder(modelConfiguration.t5Config)
        self.clipEncoder = CLIPEncoder(modelConfiguration.clipConfig)
    }
    
    public func loadWeights(from directory: URL, dtype: DType = .float16) throws {
        self.modelDirectory = directory
        logger.info("Loading weights from: \(directory.path)")
        logger.info("Using dtype: \(dtype)")
        
        try loadTransformerWeights(from: directory.appending(path: "transformer"), dtype: dtype)
        try loadVAEWeights(from: directory.appending(path: "vae"), dtype: dtype)
        try loadT5EncoderWeights(from: directory.appending(path: "text_encoder_2"), dtype: dtype)
        try loadCLIPEncoderWeights(from: directory.appending(path: "text_encoder"), dtype: dtype)
        
        logger.info("All weights loaded successfully")
    }
    
    
    private func loadTransformerWeights(from directory: URL, dtype: DType) throws {
        var transformerWeights = [String: MLXArray]()
        
        guard let enumerator = FileManager.default.enumerator(
            at: directory, includingPropertiesForKeys: nil
        ) else {
            throw FluxError.weightsNotFound("Unable to enumerate transformer directory: \(directory)")
        }
        
        for case let url as URL in enumerator {
            if url.pathExtension == "safetensors" {
                let w = try loadArrays(url: url)
                for (key, value) in w {
                    let newKey = FLUX.remapWeightKey(key)
                    if value.dtype != .bfloat16 {
                        transformerWeights[newKey] = value.asType(dtype)
                    } else {
                        transformerWeights[newKey] = value
                    }
                }
            }
        }
        transformer.update(parameters: ModuleParameters.unflattened(transformerWeights))
    }
    
    private func loadVAEWeights(from directory: URL, dtype: DType) throws {
        let vaeURL = directory.appending(path: "diffusion_pytorch_model.safetensors")
        var vaeWeights = try loadArrays(url: vaeURL)
        
        for (key, value) in vaeWeights {
            if value.dtype != .bfloat16 {
                vaeWeights[key] = value.asType(dtype)
            }
            if value.ndim == 4 {
                vaeWeights[key] = value.transposed(0, 2, 3, 1)
            }
        }
        vae.update(parameters: ModuleParameters.unflattened(vaeWeights))
    }
    
    private func loadT5EncoderWeights(from directory: URL, dtype: DType) throws {
        var weights = [String: MLXArray]()
        
        guard let enumerator = FileManager.default.enumerator(
            at: directory, includingPropertiesForKeys: nil
        ) else {
            throw FluxError.weightsNotFound("Unable to enumerate T5 encoder directory: \(directory)")
        }
        
        for case let url as URL in enumerator {
            if url.pathExtension == "safetensors" {
                let w = try loadArrays(url: url)
                for (key, value) in w {
                    if value.dtype != .bfloat16 {
                        weights[key] = value.asType(dtype)
                    } else {
                        weights[key] = value
                    }
                }
            }
        }
        
        if let relativeAttentionBias = weights[
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        ] {
            weights["relative_attention_bias.weight"] = relativeAttentionBias
        }
        
        t5Encoder.update(parameters: ModuleParameters.unflattened(weights))
    }
    
    private func loadCLIPEncoderWeights(from directory: URL, dtype: DType) throws {
        let weightsURL = directory.appending(path: "model.safetensors")
        var weights = try loadArrays(url: weightsURL)
        
        for (key, value) in weights {
            if value.dtype != .bfloat16 {
                weights[key] = value.asType(dtype)
            }
        }
        clipEncoder.update(parameters: ModuleParameters.unflattened(weights))
    }
    
    
    public func conditionText(prompt: String) -> (MLXArray, MLXArray) {
        let t5Tokens = t5Tokenizer.encode(text: prompt, addSpecialTokens: true)
        let paddedT5Tokens = Array(t5Tokens.prefix(configuration.t5MaxSequenceLength))
            + Array(repeating: 0, count: max(0, configuration.t5MaxSequenceLength - min(t5Tokens.count, configuration.t5MaxSequenceLength)))
        
        let clipTokens = clipTokenizer.tokenize(text: prompt)
        let paddedClipTokens = Array(clipTokens.prefix(configuration.clipMaxSequenceLength))
            + Array(repeating: configuration.clipPaddingToken, count: max(0, configuration.clipMaxSequenceLength - min(clipTokens.count, configuration.clipMaxSequenceLength)))
        
        let promptEmbeddings = t5Encoder(MLXArray(paddedT5Tokens)[.newAxis])
        let pooledPromptEmbeddings = clipEncoder(MLXArray(paddedClipTokens)[.newAxis])
        
        return (promptEmbeddings, pooledPromptEmbeddings)
    }
    
    
    public func ensureLoaded() {
        eval(transformer, t5Encoder, clipEncoder)
    }
    
    public func decode(xt: MLXArray) -> MLXArray {
        var x = vae.decode(xt)
        x = clip(x / 2 + 0.5, min: 0, max: 1)
        return x
    }
    
    public func detachedDecoder() -> ImageDecoder {
        let autoencoder = self.vae
        func decode(xt: MLXArray) -> MLXArray {
            var x = autoencoder.decode(xt)
            x = clip(x / 2 + 0.5, min: 0, max: 1)
            return x
        }
        return decode(xt:)
    }
}


extension FluxModelCore: FLUXComponents {}