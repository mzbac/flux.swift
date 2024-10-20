import Foundation
import Hub
import MLX
import MLXNN
import MLXRandom
import Tokenizers

open class FLUX {

  internal func loadLoraWeights(hub: HubApi, loraPath: String, dType: DType) throws
    -> [String: MLXArray]
  {
    let loraDirectory: URL
    if FileManager.default.fileExists(atPath: loraPath) {
      loraDirectory = URL(fileURLWithPath: loraPath)
    } else {
      let repo = Hub.Repo(id: loraPath)
      loraDirectory = hub.localRepoLocation(repo)
    }

    return try Self.loadLoraWeights(directory: loraDirectory, dType: dType)
  }

  internal static func loadLoraWeights(directory: URL, dType: DType) throws -> [String: MLXArray] {
    var loraWeights = [String: MLXArray]()
    let enumerator = FileManager.default.enumerator(
      at: directory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
      if url.pathExtension == "safetensors" {
        let w = try loadArrays(url: url)
        for (key, value) in w {
          let newKey = remapWeightKey(key)
          if value.dtype != .bfloat16 {
            loraWeights[newKey] = value.asType(dType)
          } else {
            loraWeights[newKey] = value
          }
        }
      }
    }
    return loraWeights
  }

  internal static func remapWeightKey(_ key: String) -> String {
    if key.contains(".ff.") || key.contains(".ff_context.") {
      let components = key.components(separatedBy: ".")
      if components.count >= 5 {
        let blockIndex = components[1]
        let ffType = components[2]  // "ff" or "ff_context"
        let netIndex = components[4]

        if netIndex == "0" {
          return "transformer_blocks.\(blockIndex).\(ffType).linear1.\(components.last!)"
        } else if netIndex == "2" {
          return "transformer_blocks.\(blockIndex).\(ffType).linear2.\(components.last!)"
        }
      }
    }
    return key
  }

  internal static func loadTokenizers(directory: URL, hub: HubApi) throws -> (
    any Tokenizer, CLIPTokenizer
  ) {
    let t5TokenizerConfig = try? hub.configuration(
      fileURL: directory.appending(path: "tokenizer_2/tokenizer_config.json"))
    let t5TokenizerVocab = try hub.configuration(
      fileURL: directory.appending(path: "tokenizer_2/tokenizer.json"))
    let t5Tokenizer = try AutoTokenizer.from(
      tokenizerConfig: t5TokenizerConfig!, tokenizerData: t5TokenizerVocab)

    let vocabulary = try JSONDecoder().decode(
      [String: Int].self, from: Data(contentsOf: directory.appending(path: "tokenizer/vocab.json"))
    )
    let merges = try String(contentsOf: directory.appending(path: "tokenizer/merges.txt"))
      .components(separatedBy: .newlines)
      .dropFirst()
      .filter { !$0.isEmpty }
    let clipTokenizer = CLIPTokenizer(merges: merges, vocabulary: vocabulary)

    return (t5Tokenizer, clipTokenizer)
  }

  internal static func loadVAE(directory: URL, dType: DType) throws -> VAE {
    let vaeConfig = VAEConfiguration()
    let vae = VAE(vaeConfig)

    var vaeWeights = try loadArrays(
      url: directory.appending(path: "vae/diffusion_pytorch_model.safetensors"))

    for (key, value) in vaeWeights {
      if value.dtype != .bfloat16 {
        vaeWeights[key] = value.asType(dType)
      }
      if value.ndim == 4 {
        vaeWeights[key] = value.transposed(0, 2, 3, 1)
      }
    }
    vae.update(parameters: ModuleParameters.unflattened(vaeWeights))
    return vae
  }

  internal static func loadT5Encoder(directory: URL, dType: DType) throws -> T5Encoder {
    let t5Encoder = T5Encoder(T5Configuration())
    var t5Weights = [String: MLXArray]()
    let t5Enumerator = FileManager.default.enumerator(
      at: directory.appending(path: "text_encoder_2"), includingPropertiesForKeys: nil)!
    for case let url as URL in t5Enumerator {
      if url.pathExtension == "safetensors" {
        let w = try loadArrays(url: url)
        for (key, value) in w {
          if value.dtype != .bfloat16 {
            t5Weights[key] = value.asType(dType)
          } else {
            t5Weights[key] = value
          }
        }
      }
    }
    // Handle relative attention bias
    if let relativeAttentionBias = t5Weights[
      "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"]
    {
      t5Weights["relative_attention_bias.weight"] = relativeAttentionBias
    }

    t5Encoder.update(parameters: ModuleParameters.unflattened(t5Weights))
    return t5Encoder
  }

  internal static func loadCLIPEncoder(directory: URL, dType: DType) throws -> CLIPEncoder {
    let clipEncoder = CLIPEncoder(CLIPConfiguration())
    var clipWeights = try loadArrays(
      url: directory.appending(path: "text_encoder/model.safetensors"))

    for (key, value) in clipWeights {
      if value.dtype != .bfloat16 {
        clipWeights[key] = value.asType(dType)
      }
    }
    clipEncoder.update(parameters: ModuleParameters.unflattened(clipWeights))
    return clipEncoder
  }
}

public class Flux1Schnell: FLUX, TextToImageGenerator, @unchecked Sendable {
  let clipTokenizer: CLIPTokenizer
  let t5Tokenizer: any Tokenizer
  public let transformer: MultiModalDiffusionTransformer
  let vae: VAE
  let t5Encoder: T5Encoder
  let clipEncoder: CLIPEncoder

  public init(hub: HubApi, configuration: FluxConfiguration, dType: DType) throws {
    let repo = Hub.Repo(id: configuration.id)
    let directory = hub.localRepoLocation(repo)

    (self.t5Tokenizer, self.clipTokenizer) = try Self.loadTokenizers(directory: directory, hub: hub)
    self.vae = try Self.loadVAE(directory: directory, dType: dType)
    self.transformer = try Self.loadTransformer(directory: directory, dType: dType)
    self.t5Encoder = try Self.loadT5Encoder(directory: directory, dType: dType)
    self.clipEncoder = try Self.loadCLIPEncoder(directory: directory, dType: dType)
  }

  private static func loadTransformer(directory: URL, dType: DType) throws
    -> MultiModalDiffusionTransformer
  {
    let transformer = MultiModalDiffusionTransformer(MultiModalDiffusionConfiguration())
    var transformerWeights = [String: MLXArray]()
    let enumerator = FileManager.default.enumerator(
      at: directory.appending(path: "transformer"), includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
      if url.pathExtension == "safetensors" {
        let w = try loadArrays(url: url)
        for (key, value) in w {
          let newKey = remapWeightKey(key)
          if value.dtype != .bfloat16 {
            transformerWeights[newKey] = value.asType(dType)
          } else {
            transformerWeights[newKey] = value
          }
        }
      }
    }
    transformer.update(parameters: ModuleParameters.unflattened(transformerWeights))
    return transformer
  }

  public func conditionText(prompt: String) -> (MLXArray, MLXArray) {
    let t5Tokens = t5Tokenizer.encode(text: prompt, addSpecialTokens: true)
    let paddedT5Tokens =
      Array(t5Tokens.prefix(256))
      + Array(repeating: 0, count: max(0, 256 - min(t5Tokens.count, 256)))
    let clipTokens = clipTokenizer.tokenize(text: prompt)
    let paddedClipTokens =
      Array(clipTokens.prefix(77))
      + Array(repeating: 49407, count: max(0, 77 - min(clipTokens.count, 77)))
    let promptEmbeddings = t5Encoder(MLXArray(paddedT5Tokens)[.newAxis])
    let pooledPromptEmbeddings = clipEncoder(MLXArray(paddedClipTokens)[.newAxis])

    return (promptEmbeddings, pooledPromptEmbeddings)
  }

  open func ensureLoaded() {
    eval(transformer, t5Encoder, clipEncoder, vae)
  }

  public func decode(xt: MLXArray) -> MLXArray {
    detachedDecoder()(xt)
  }
  public func detachedDecoder() -> ImageDecoder {
    let autoencoder = self.vae
    func decode(xt: MLXArray) -> MLXArray {
      var x = autoencoder.decode(latents: xt)
      x = clip(x / 2 + 0.5, min: 0, max: 1)
      return x
    }
    return decode(xt:)
  }
}

public class Flux1Dev: FLUX, TextToImageGenerator, @unchecked Sendable {
  let clipTokenizer: CLIPTokenizer
  let t5Tokenizer: any Tokenizer
  public let transformer: MultiModalDiffusionTransformer
  let vae: VAE
  let t5Encoder: T5Encoder
  let clipEncoder: CLIPEncoder

  public init(hub: HubApi, configuration: FluxConfiguration, dType: DType) throws {
    let repo = Hub.Repo(id: configuration.id)
    let directory = hub.localRepoLocation(repo)

    (self.t5Tokenizer, self.clipTokenizer) = try Self.loadTokenizers(directory: directory, hub: hub)
    self.vae = try Self.loadVAE(directory: directory, dType: dType)
    self.transformer = try Self.loadTransformer(directory: directory, dType: dType)
    self.t5Encoder = try Self.loadT5Encoder(directory: directory, dType: dType)
    self.clipEncoder = try Self.loadCLIPEncoder(directory: directory, dType: dType)
  }

  private static func loadTransformer(directory: URL, dType: DType) throws
    -> MultiModalDiffusionTransformer
  {
    let transformer = MultiModalDiffusionTransformer(
      MultiModalDiffusionConfiguration(guidanceEmbeds: true))
    var transformerWeights = [String: MLXArray]()
    let enumerator = FileManager.default.enumerator(
      at: directory.appending(path: "transformer"), includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
      if url.pathExtension == "safetensors" {
        let w = try loadArrays(url: url)
        for (key, value) in w {
          let newKey = remapWeightKey(key)
          if value.dtype != .bfloat16 {
            transformerWeights[newKey] = value.asType(dType)
          } else {
            transformerWeights[newKey] = value
          }
        }
      }
    }
    transformer.update(parameters: ModuleParameters.unflattened(transformerWeights))
    return transformer
  }

  public func conditionText(prompt: String) -> (MLXArray, MLXArray) {
    let t5Tokens = t5Tokenizer.encode(text: prompt, addSpecialTokens: true)
    let paddedT5Tokens =
      Array(t5Tokens.prefix(512))
      + Array(repeating: 0, count: max(0, 512 - min(t5Tokens.count, 512)))
    let clipTokens = clipTokenizer.tokenize(text: prompt)
    let paddedClipTokens =
      Array(clipTokens.prefix(77))
      + Array(repeating: 49407, count: max(0, 77 - min(clipTokens.count, 77)))
    let promptEmbeddings = t5Encoder(MLXArray(paddedT5Tokens)[.newAxis])
    let pooledPromptEmbeddings = clipEncoder(MLXArray(paddedClipTokens)[.newAxis])

    return (promptEmbeddings, pooledPromptEmbeddings)
  }

  open func ensureLoaded() {
    eval(transformer, t5Encoder, clipEncoder, vae)
  }

  public func decode(xt: MLXArray) -> MLXArray {
    detachedDecoder()(xt)
  }
  public func detachedDecoder() -> ImageDecoder {
    let autoencoder = self.vae
    func decode(xt: MLXArray) -> MLXArray {
      var x = autoencoder.decode(latents: xt)
      x = clip(x / 2 + 0.5, min: 0, max: 1)
      return x
    }
    return decode(xt:)
  }
}
public protocol ImageGenerator {
  func ensureLoaded()

  /// Return a detached decoder -- this is useful if trying to conserve memory.
  ///
  /// The decoder can be used independently of the ImageGenerator to transform
  /// latents into raster images.
  func detachedDecoder() -> ImageDecoder

  /// the equivalent to the ``detachedDecoder()`` but without the detatching
  func decode(xt: MLXArray) -> MLXArray
}

public protocol TextToImageGenerator: ImageGenerator, Sendable {
    var transformer: MultiModalDiffusionTransformer { get }
    func conditionText(prompt: String) -> (MLXArray, MLXArray)
}

extension TextToImageGenerator {
    public func generateLatents(parameters: EvaluateParameters) -> DenoiseIterator {
        let latentsShape = [1, (parameters.height / 16) * (parameters.width / 16), 64]
        let latents: MLXArray
        if let seed = parameters.seed {
            latents = MLXRandom.normal(latentsShape, key: MLXRandom.key(seed))
        } else {
            latents = MLXRandom.normal(latentsShape)
        }
        let (promptEmbeddings, pooledPromptEmbeddings) = conditionText(prompt: parameters.prompt)

        return DenoiseIterator(
            steps: parameters.numInferenceSteps,
            promptEmbeddings: promptEmbeddings,
            pooledPromptEmbeddings: pooledPromptEmbeddings,
            latents: latents,
            evaluateParameters: parameters,
            transformer: transformer
        )
    }
}

/// Public interface for transforming a text prompt into an image.
///
/// Steps:
///
/// - ``generateLatents(image:parameters:strength:)``
/// - evaluate each of the latents from the iterator
/// - ``ImageGenerator/decode(xt:)`` or ``ImageGenerator/detachedDecoder()`` to convert the final latent into an image
/// - use ``Image`` to save the image
public protocol ImageToImageGenerator: ImageGenerator, Sendable {
    func generateLatents(image: MLXArray, parameters: EvaluateParameters, strength: Float)
        -> DenoiseIterator
}

public typealias ImageDecoder = (MLXArray) -> MLXArray

public struct DenoiseIterator: Sequence, IteratorProtocol {
  let steps: Int
  let promptEmbeddings: MLXArray
  let pooledPromptEmbeddings: MLXArray
  var latents: MLXArray
  public var i: Int
  let evaluateParameters: EvaluateParameters
  let transformer: MultiModalDiffusionTransformer

  init(
    startStep: Int = 0, steps: Int, promptEmbeddings: MLXArray, pooledPromptEmbeddings: MLXArray,
    latents: MLXArray,
    evaluateParameters: EvaluateParameters, transformer: MultiModalDiffusionTransformer
  ) {
    self.steps = steps
    self.promptEmbeddings = promptEmbeddings
    self.pooledPromptEmbeddings = pooledPromptEmbeddings
    self.latents = latents
    self.i = startStep
    self.evaluateParameters = evaluateParameters
    self.transformer = transformer
  }

  public mutating func next() -> MLXArray? {
    guard i < steps else {
      return nil
    }
    let noise = transformer(
      t: i,
      promptEmbeds: promptEmbeddings,
      pooledPromptEmbeds: pooledPromptEmbeddings,
      hiddenStates: latents,
      evaluateParameters: evaluateParameters
    )

    let dt = evaluateParameters.sigmas[i + 1] - evaluateParameters.sigmas[i]
    latents += noise * dt
    i += 1
    return latents
  }
}