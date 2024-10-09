import Foundation
import Hub
import MLX
import MLXNN
import MLXRandom
import Tokenizers

open class FLUX {

}

public class Flux1Schnell: FLUX, TextToImageGenerator {
  let clipTokenizer: CLIPTokenizer
  let t5Tokenizer: any Tokenizer
  let vae: VAE
  let transformer: MultiModalDiffusionTransformer
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

  private static func loadTokenizers(directory: URL, hub: HubApi) throws -> (
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

  private static func loadVAE(directory: URL, dType: DType) throws -> VAE {
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
          if value.dtype != .bfloat16 {
            transformerWeights[key] = value.asType(dType)

          } else {
            transformerWeights[key] = value
          }
        }
      }
    }
    transformer.update(parameters: ModuleParameters.unflattened(transformerWeights))
    return transformer
  }

  private static func loadT5Encoder(directory: URL, dType: DType) throws -> T5Encoder {
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
      for i in 1..<32 {  // Assuming there are 32 blocks, adjust if necessary
        let key = "encoder.block.\(i).layer.0.SelfAttention.relative_attention_bias.weight"
        t5Weights[key] = relativeAttentionBias
      }
    }
    t5Encoder.update(parameters: ModuleParameters.unflattened(t5Weights))
    return t5Encoder
  }

  private static func loadCLIPEncoder(directory: URL, dType: DType) throws -> CLIPEncoder {
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

  public func generateLatents(parameters: EvaluateParameters) -> DenoiseIterator {
    let latentsShape = [1, (parameters.height / 16) * (parameters.width / 16), 64]
    let latents = MLXRandom.normal(latentsShape, key: MLXRandom.key(parameters.seed))
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

  func conditionText(prompt: String) -> (MLXArray, MLXArray) {
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

public protocol TextToImageGenerator: ImageGenerator {
  func generateLatents(parameters: EvaluateParameters) -> DenoiseIterator
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
    steps: Int, promptEmbeddings: MLXArray, pooledPromptEmbeddings: MLXArray, latents: MLXArray,
    evaluateParameters: EvaluateParameters, transformer: MultiModalDiffusionTransformer
  ) {
    self.steps = steps
    self.promptEmbeddings = promptEmbeddings
    self.pooledPromptEmbeddings = pooledPromptEmbeddings
    self.latents = latents
    self.i = 0
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
