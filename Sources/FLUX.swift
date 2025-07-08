import Foundation
import Hub
import MLX
import MLXNN
import MLXRandom
import Tokenizers
import Logging

private let logger = Logger(label: "flux.swift.FLUX")

open class FLUX {
  public var modelDirectory: URL?
  
 
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
    
    guard let enumerator = FileManager.default.enumerator(
      at: directory, includingPropertiesForKeys: nil
    ) else {
      throw FluxError.weightsNotFound("Unable to enumerate LoRA directory: \(directory)")
    }
    
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
        let ffType = components[2]
        let netIndex = components[4]
        
        if netIndex == "0" {
          return "transformer_blocks.\(blockIndex).\(ffType).linear1.\(components.last ?? "")"
        } else if netIndex == "2" {
          return "transformer_blocks.\(blockIndex).\(ffType).linear2.\(components.last ?? "")"
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
    
    guard let t5Config = t5TokenizerConfig else {
      throw FluxError.weightsNotFound("T5 tokenizer configuration not found")
    }
    
    let t5Tokenizer = try AutoTokenizer.from(
      tokenizerConfig: t5Config, tokenizerData: t5TokenizerVocab)
    
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
}


public class Flux1Schnell: FLUX, TextToImageGenerator, FLUXComponents, @unchecked Sendable {
  private let core: FluxModelCore
  
  public var transformer: MultiModalDiffusionTransformer { core.transformer }
  public var vae: VAE { core.vae }
  public var t5Encoder: T5Encoder { core.t5Encoder }
  public var clipEncoder: CLIPEncoder { core.clipEncoder }
  
  public init(hub: HubApi, configuration: FluxConfiguration) throws {
    self.core = try FluxModelCore(
      hub: hub,
      fluxConfiguration: configuration,
      modelConfiguration: .schnell
    )
    super.init()
  }
  
  public init(hub: HubApi, modelDirectory: URL) throws {
    self.core = try FluxModelCore(
      hub: hub,
      modelDirectory: modelDirectory,
      modelConfiguration: .schnell
    )
    super.init()
  }
  
  public convenience init(hub: HubApi, configuration: FluxConfiguration, dType: DType) throws {
    try self.init(hub: hub, configuration: configuration)
    let repo = Hub.Repo(id: configuration.id)
    let directory = hub.localRepoLocation(repo)
    try self.loadWeights(from: directory, dtype: dType)
  }
  
  public static func createAndLoad(hub: HubApi, configuration: FluxConfiguration, dtype: DType = .float16) throws -> Flux1Schnell {
    let model = try Flux1Schnell(hub: hub, configuration: configuration)
    let repo = Hub.Repo(id: configuration.id)
    let directory = hub.localRepoLocation(repo)
    try model.loadWeights(from: directory, dtype: dtype)
    return model
  }
  
  public func loadWeights(from directory: URL, dtype: DType = .float16) throws {
    self.modelDirectory = directory
    try core.loadWeights(from: directory, dtype: dtype)
  }
  
  public func conditionText(prompt: String) -> (MLXArray, MLXArray) {
    core.conditionText(prompt: prompt)
  }
  
  public func ensureLoaded() {
    core.ensureLoaded()
  }
  
  public func decode(xt: MLXArray) -> MLXArray {
    core.decode(xt: xt)
  }
  
  public func detachedDecoder() -> ImageDecoder {
    core.detachedDecoder()
  }
}

public class Flux1Dev: FLUX, TextToImageGenerator, FLUXComponents, @unchecked Sendable {
  private let core: FluxModelCore
  
  public var transformer: MultiModalDiffusionTransformer { core.transformer }
  public var vae: VAE { core.vae }
  public var t5Encoder: T5Encoder { core.t5Encoder }
  public var clipEncoder: CLIPEncoder { core.clipEncoder }
  
  public init(hub: HubApi, configuration: FluxConfiguration) throws {
    self.core = try FluxModelCore(
      hub: hub,
      fluxConfiguration: configuration,
      modelConfiguration: .dev
    )
    super.init()
  }
  
  public init(hub: HubApi, modelDirectory: URL) throws {
    self.core = try FluxModelCore(
      hub: hub,
      modelDirectory: modelDirectory,
      modelConfiguration: .dev
    )
    super.init()
  }
  
  public convenience init(hub: HubApi, configuration: FluxConfiguration, dType: DType) throws {
    try self.init(hub: hub, configuration: configuration)
    let repo = Hub.Repo(id: configuration.id)
    let directory = hub.localRepoLocation(repo)
    try self.loadWeights(from: directory, dtype: dType)
  }
  
  public static func createAndLoad(hub: HubApi, configuration: FluxConfiguration, dtype: DType = .float16) throws -> Flux1Dev {
    let model = try Flux1Dev(hub: hub, configuration: configuration)
    let repo = Hub.Repo(id: configuration.id)
    let directory = hub.localRepoLocation(repo)
    try model.loadWeights(from: directory, dtype: dtype)
    return model
  }
  
  public func loadWeights(from directory: URL, dtype: DType = .float16) throws {
    self.modelDirectory = directory
    try core.loadWeights(from: directory, dtype: dtype)
  }
  
  public func conditionText(prompt: String) -> (MLXArray, MLXArray) {
    core.conditionText(prompt: prompt)
  }
  
  public func ensureLoaded() {
    core.ensureLoaded()
  }
  
  public func decode(xt: MLXArray) -> MLXArray {
    core.decode(xt: xt)
  }
  
  public func detachedDecoder() -> ImageDecoder {
    core.detachedDecoder()
  }
}

public class Flux1KontextDev: FLUX, TextToImageGenerator, KontextImageToImageGenerator, FLUXComponents, @unchecked Sendable {
  private let core: FluxModelCore
  
  public var transformer: MultiModalDiffusionTransformer { core.transformer }
  public var vae: VAE { core.vae }
  public var t5Encoder: T5Encoder { core.t5Encoder }
  public var clipEncoder: CLIPEncoder { core.clipEncoder }
  
  public init(hub: HubApi, configuration: FluxConfiguration) throws {
    self.core = try FluxModelCore(
      hub: hub,
      fluxConfiguration: configuration,
      modelConfiguration: .kontextDev
    )
    super.init()
  }
  
  public init(hub: HubApi, modelDirectory: URL) throws {
    self.core = try FluxModelCore(
      hub: hub,
      modelDirectory: modelDirectory,
      modelConfiguration: .kontextDev
    )
    super.init()
  }
  
  public convenience init(hub: HubApi, configuration: FluxConfiguration, dType: DType) throws {
    try self.init(hub: hub, configuration: configuration)
    let repo = Hub.Repo(id: configuration.id)
    let directory = hub.localRepoLocation(repo)
    try self.loadWeights(from: directory, dtype: dType)
  }
  
  public func loadWeights(from directory: URL, dtype: DType = .float16) throws {
    self.modelDirectory = directory
    try core.loadWeights(from: directory, dtype: dtype)
  }
  
  public func conditionText(prompt: String) -> (MLXArray, MLXArray) {
    core.conditionText(prompt: prompt)
  }
  
  public func ensureLoaded() {
    core.ensureLoaded()
  }
  
  public func decode(xt: MLXArray) -> MLXArray {
    core.decode(xt: xt)
  }
  
  public func detachedDecoder() -> ImageDecoder {
    core.detachedDecoder()
  }
}

public protocol ImageGenerator {
  func ensureLoaded()
  func detachedDecoder() -> ImageDecoder
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

public protocol ImageToImageGenerator: ImageGenerator, Sendable {
  var transformer: MultiModalDiffusionTransformer { get }
  var vae: VAE { get }
  func conditionText(prompt: String) -> (MLXArray, MLXArray)
  func generateLatents(image: MLXArray, parameters: EvaluateParameters, strength: Float)
    -> DenoiseIterator
}

extension ImageToImageGenerator {
  internal func packLatents(latents: MLXArray, height: Int, width: Int) -> MLXArray {
    let reshaped = latents.reshaped(1, height / 16, 2, width / 16, 2, 16)
    let transposed = reshaped.transposed(0, 1, 3, 5, 2, 4)
    return transposed.reshaped(1, (height / 16) * (width / 16), 64)
  }
  
  public func generateLatents(image: MLXArray, parameters: EvaluateParameters, strength: Float)
    -> DenoiseIterator
  {
    if let seed = parameters.seed {
      MLXRandom.seed(seed)
    }
    let noise = MLXRandom.normal([1, (parameters.height / 16) * (parameters.width / 16), 64])
    
    let strength = max(0.0, min(1.0, strength))
    
    let startStep = max(1, Int(Float(parameters.numInferenceSteps) * strength))
    
    var latents = vae.encode(image[.newAxis])
    
    latents = packLatents(latents: latents, height: parameters.height, width: parameters.width)
    
    let sigma = parameters.sigmas[startStep]
    
    latents = (latents * (1 - sigma) + sigma * noise)
    
    let (promptEmbeddings, pooledPromptEmbeddings) = conditionText(prompt: parameters.prompt)
    
    return DenoiseIterator(
      startStep: startStep,
      steps: parameters.numInferenceSteps,
      promptEmbeddings: promptEmbeddings,
      pooledPromptEmbeddings: pooledPromptEmbeddings,
      latents: latents,
      evaluateParameters: parameters,
      transformer: transformer
    )
  }
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

// MARK: - Quantization Extensions

public extension FLUX {
  func saveQuantizedWeights(to path: URL, bits: Int = 4, groupSize: Int = 64) throws {
    guard let fluxModel = self as? FLUXComponents else {
      throw FluxError.modelComponentMissing
    }
    
    logger.info("Saving quantized weights to: \(path.path)")
    logger.info("Quantization parameters: \(bits)-bit, group size: \(groupSize)")
    logger.info("Model type: \(String(describing: type(of: self)))")
    
    if !fluxModel.hasQuantizedLayers {
      logger.info("Model is not quantized. Quantizing before saving...")
      fluxModel.quantizeAllComponents(bits: bits, groupSize: groupSize)
    } else {
      let currentQuantization = detectCurrentQuantization(fluxModel)
      if let current = currentQuantization {
        if current.bits != bits || current.groupSize != groupSize {
          logger.error("Model is already quantized with different parameters")
          logger.error("Current: \(current.bits)-bit, group size: \(current.groupSize)")
          logger.error("Requested: \(bits)-bit, group size: \(groupSize)")
          throw FluxError.quantizationMismatch(
            current: "\(current.bits)-bit, group size: \(current.groupSize)",
            requested: "\(bits)-bit, group size: \(groupSize)"
          )
        }
      }
      logger.info("Model is already quantized with matching parameters, saving current state...")
      logger.info("Found \(fluxModel.quantizedLayerCount) quantized layers")
    }
    
    try QuantizationUtils.saveQuantizedModel(
      fluxModel,
      to: path,
      quantizationBits: bits,
      groupSize: groupSize,
      modelDirectory: self.modelDirectory
    )
    
    logger.info("Quantized weights saved successfully!")
  }
  
  var isQuantized: Bool {
    return hasQuantizedLayers()
  }
  
  private func detectCurrentQuantization(_ fluxModel: FLUXComponents) -> (bits: Int, groupSize: Int)? {
    for component in fluxModel.components {
      let leafModules = component.module.leafModules().flattened()
      for (_, module) in leafModules {
        if let quantizedLinear = module as? QuantizedLinear {
          return (bits: quantizedLinear.bits, groupSize: quantizedLinear.groupSize)
        }
      }
    }
    return nil
  }
  
  func hasQuantizedLayers() -> Bool {
    if let fluxModel = self as? FLUXComponents {
      return fluxModel.hasQuantizedLayers
    }
    return false
  }
}

// MARK: - Loading Quantized Models

public extension FLUX {
  static func loadQuantized(
    from source: String,
    modelType: String = "schnell",
    hub: HubApi = HubApi(),
    configuration: LoadConfiguration = LoadConfiguration(),
    progressHandler: ((Progress) -> Void)? = nil
  ) async throws -> FLUX {
    logger.info("Loading pre-quantized model from: \(source)")
    
    let modelPath: String
    
    if FileManager.default.fileExists(atPath: source) {
      logger.info("Using local path: \(source)")
      modelPath = source
    } else {
      logger.info("Treating as Hugging Face model ID: \(source)")
      logger.info("Downloading quantized model...")
      
      let repo = Hub.Repo(id: source)
      let localURL = try await hub.snapshot(
        from: repo,
        matching: ["**/*.safetensors", "**/*.json", "**/*.txt", "metadata.json", "README.md", "**/spiece.model"]
      ) { progress in
        let percent = Int(progress.fractionCompleted * 100)
        if percent % 10 == 0 {
          logger.info("Download progress: \(percent)%")
        }
        progressHandler?(progress)
      }
      
      modelPath = localURL.path
      logger.info("Downloaded to: \(modelPath)")
      
      if let contents = try? FileManager.default.contentsOfDirectory(atPath: modelPath) {
        logger.info("Downloaded files: \(contents)")
      }
    }
    
    logger.info("Loading quantized weights and metadata...")
    
    let (loadedWeights, info) = try QuantizationUtils.loadQuantizedModel(from: URL(fileURLWithPath: modelPath))
    
    guard info.isQuantized,
          let bits = info.quantizationBits,
          let groupSize = info.groupSize else {
      throw FluxError.quantizationFailed("Model is not quantized")
    }
    
    logger.info("Loaded \(loadedWeights.count) weights")
    logger.info("Quantization: \(bits)-bit, group size: \(groupSize)")
    
    // Pass the quantized model directory so tokenizers are loaded from there
    let modelDirectory = URL(fileURLWithPath: modelPath)
    let model = try createModel(type: modelType, hub: hub, modelDirectory: modelDirectory)
    
    if let fluxModel = model as? FLUXComponents {
      logger.info("Applying selective quantization...")
      fluxModel.applySelectiveQuantization(weights: loadedWeights, bits: bits, groupSize: groupSize)
    } else {
      throw FluxError.modelComponentMissing
    }
    
    logger.info("Loading weights into quantized model...")
    if let fluxModel = model as? FLUXComponents {
      fluxModel.updateComponentWeights(from: loadedWeights)
    }
    
    if let loraPath = configuration.loraPath {
      logger.info("Applying LoRA weights from: \(loraPath)")
      do {
        let loraWeights = try model.loadLoraWeights(hub: hub, loraPath: loraPath, dType: configuration.dType)
        
        if let fluxModel = model as? FLUXComponents {
          applyLoraWeights(to: fluxModel.transformer, loraWeight: loraWeights)
          logger.info("LoRA weights applied successfully to quantized model")
        }
      } catch {
        logger.error("Failed to apply LoRA weights: \(error)")
        throw error
      }
    }
    
    logger.info("Pre-quantized model loaded successfully!")
    return model
  }
  
  private static func createModel(type: String, hub: HubApi, modelDirectory: URL? = nil) throws -> FLUX {
    logger.info("Creating model structure for type: \(type)")
    
    if let modelDirectory = modelDirectory {
      logger.info("Using custom model directory: \(modelDirectory.path)")
      switch type.lowercased() {
      case "schnell":
        return try Flux1Schnell(hub: hub, modelDirectory: modelDirectory)
      case "dev":
        return try Flux1Dev(hub: hub, modelDirectory: modelDirectory)
      case "kontext":
        return try Flux1KontextDev(hub: hub, modelDirectory: modelDirectory)
      default:
        logger.info("Unknown model type: \(type), defaulting to schnell")
        return try Flux1Schnell(hub: hub, modelDirectory: modelDirectory)
      }
    } else {
      switch type.lowercased() {
      case "schnell":
        return try Flux1Schnell(hub: hub, configuration: FluxConfiguration.schnell())
      case "dev":
        return try Flux1Dev(hub: hub, configuration: FluxConfiguration.dev())
      case "kontext":
        return try Flux1KontextDev(hub: hub, configuration: FluxConfiguration.flux1kontextDev())
      default:
        logger.info("Unknown model type: \(type), defaulting to schnell")
        return try Flux1Schnell(hub: hub, configuration: FluxConfiguration.schnell())
      }
    }
  }
}


public enum FluxError: LocalizedError {
  case modelComponentMissing
  case weightsNotFound(String)
  case quantizationFailed(String)
  case quantizationMismatch(current: String, requested: String)
  
  public var errorDescription: String? {
    switch self {
    case .modelComponentMissing:
      return "Required model component not found"
    case .weightsNotFound(let path):
      return "Weights not found at: \(path)"
    case .quantizationFailed(let reason):
      return "Quantization failed: \(reason)"
    case .quantizationMismatch(let current, let requested):
      return "Cannot re-quantize an already quantized model. Current: \(current), Requested: \(requested). Please load an unquantized model to apply different quantization parameters."
    }
  }
}