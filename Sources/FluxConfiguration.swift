import Foundation
import Hub
import MLX
import MLXNN
import Logging

private let logger = Logger(label: "flux.swift.FluxConfiguration")

public struct LoadConfiguration: Sendable {

  public var float16 = true

  public var quantize = false

  public let loraPath: String?
  public var dType: DType {
    float16 ? .float16 : .float32
  }

  public init(
    float16: Bool = true, quantize: Bool = false, loraPath: String? = nil
  ) {
    self.float16 = float16
    self.quantize = quantize
    self.loraPath = loraPath
  }
}

public struct EvaluateParameters {
  public var width: Int
  public var height: Int
  public var numInferenceSteps: Int
  public var guidance: Float
  public var seed: UInt64?
  public var prompt: String
  public var numTrainSteps: Int
  public let sigmas: MLXArray
  public let dimensionsExplicitlySet: Bool

  public init(
    width: Int = 512,
    height: Int = 512,
    numInferenceSteps: Int = 4,
    guidance: Float = 4.0,
    seed: UInt64? = nil,
    prompt: String = "",
    numTrainSteps: Int = 1000,
    shiftSigmas: Bool = false,
    dimensionsExplicitlySet: Bool = false
  ) {
    if width % 16 != 0 || height % 16 != 0 {
      logger.warning("Width and height should be multiples of 16. Rounding down.")
    }
    self.width = 16 * (width / 16)
    self.height = 16 * (height / 16)
    self.numInferenceSteps = numInferenceSteps
    self.guidance = guidance
    self.seed = seed
    self.prompt = prompt
    self.numTrainSteps = numTrainSteps
    self.dimensionsExplicitlySet = dimensionsExplicitlySet
    self.sigmas = Self.createSigmasValues(
      numInferenceSteps: numInferenceSteps, shiftSigmas: shiftSigmas, width: width, height: height)
  }

  private static func createSigmasValues(
    numInferenceSteps: Int, shiftSigmas: Bool = false, width: Int = 512, height: Int = 512
  ) -> MLXArray {
    var sigmas = MLXArray.linspace(1, 1.0 / Float(numInferenceSteps), count: numInferenceSteps)
    if shiftSigmas {
      let y1: Float = 0.5
      let x1: Float = 256
      let m = (1.5 - y1) / (4096 - x1)
      let b = y1 - m * x1
      let mu = m * Float(width * height) / 256 + b
      let muArray = MLXArray(mu)
      let shiftedSigmas = MLX.exp(muArray) / (MLX.exp(muArray) + (1 / sigmas - 1))
      sigmas = shiftedSigmas
    }
    sigmas = MLX.concatenated([sigmas, MLXArray.zeros([1])])
    return sigmas
  }
}

enum FileKey {
  case mmditWeights
  case textEncoderWeights
  case textEncoderWeights2
  case vaeWeights
  case tokenizer
  case tokenizer2
  case modelIndex
}

func applyLoraWeights(
  to transform: Module, loraWeight: [String: MLXArray], loraScale: Float = 1.0
) {
  var layerUpdates: [String: MLXArray] = [:]
  
  for (key, module) in transform.namedModules() {
    let loraAKey = "transformer." + key + ".lora_A.weight"
    let loraBKey = "transformer." + key + ".lora_B.weight"
    
    if let loraA = loraWeight[loraAKey], let loraB = loraWeight[loraBKey] {
      if let quantizedLinear = module as? QuantizedLinear {
        logger.info("Applying LoRA to quantized layer: \(key)")
        
        let dequantizedWeight = dequantized(
          quantizedLinear.weight,
          scales: quantizedLinear.scales,
          biases: quantizedLinear.biases,
          groupSize: quantizedLinear.groupSize,
          bits: quantizedLinear.bits
        )
        
        let loraDelta = matmul(loraB, loraA)
        let fusedWeight = dequantizedWeight + loraScale * loraDelta
        
        let fusedLinear = Linear(
          weight: fusedWeight,
          bias: quantizedLinear.bias
        )
        
        let requantized = QuantizedLinear(
          fusedLinear,
          groupSize: quantizedLinear.groupSize,
          bits: quantizedLinear.bits
        )
        
        layerUpdates[key + ".weight"] = requantized.weight
        layerUpdates[key + ".scales"] = requantized.scales
        layerUpdates[key + ".biases"] = requantized.biases
        
      } else if let linear = module as? Linear {
        logger.info("Fusing LoRA weights into linear layer: \(key)")
        let loraDelta = matmul(loraB, loraA)
        let currentWeight = linear.weight
        let newWeight = currentWeight + loraScale * loraDelta
        layerUpdates[key + ".weight"] = newWeight
      }
    }
  }
  
  if !layerUpdates.isEmpty {
    transform.update(parameters: ModuleParameters.unflattened(layerUpdates))
  }
}

public struct FluxConfiguration: Sendable {
  public var id: String
  let files: [FileKey: String]
  public let defaultParameters: @Sendable () -> EvaluateParameters
  let factory:
    @Sendable (HubApi, FluxConfiguration, LoadConfiguration) throws ->
      FLUX

  public func download(
    hub: HubApi = HubApi(), progressHandler: @escaping (Progress) -> Void = { _ in }
  ) async throws {
    let repo = Hub.Repo(id: self.id)
    try await hub.snapshot(
      from: repo, matching: Array(files.values), progressHandler: progressHandler)
  }

  public func downloadLoraWeights(
    hub: HubApi = HubApi(), loadConfiguration: LoadConfiguration,
    progressHandler: @escaping (Progress) -> Void = { _ in }
  ) async throws {
    guard let loraPath = loadConfiguration.loraPath else {
      throw FluxConfigurationError.missingLoraPath
    }

    let repo = Hub.Repo(id: loraPath)
    try await hub.snapshot(
      from: repo, matching: ["*.safetensors"], progressHandler: progressHandler)
  }

  public func textToImageGenerator(hub: HubApi = HubApi(), configuration: LoadConfiguration)
    throws -> TextToImageGenerator?
  {
    try factory(hub, self, configuration) as? TextToImageGenerator
  }

  public func ImageToImageGenerator(hub: HubApi = HubApi(), configuration: LoadConfiguration)
    throws -> ImageToImageGenerator?
  {
    try factory(hub, self, configuration) as? ImageToImageGenerator
  }

  public func kontextImageToImageGenerator(hub: HubApi = HubApi(), configuration: LoadConfiguration)
    throws -> KontextImageToImageGenerator?
  {
    let generator = try factory(hub, self, configuration) as? KontextImageToImageGenerator
    return generator
  }

  public static let flux1Schnell = FluxConfiguration(
    id: "black-forest-labs/FLUX.1-schnell",
    files: [
      .mmditWeights: "transformer/*.safetensors",
      .textEncoderWeights: "text_encoder/model.safetensors",
      .textEncoderWeights2: "text_encoder_2/*.safetensors",
      .tokenizer: "tokenizer/*",
      .tokenizer2: "tokenizer_2/*",
      .vaeWeights: "vae/diffusion_pytorch_model.safetensors",
      .modelIndex: "model_index.json",
    ],
    defaultParameters: { EvaluateParameters() },
    factory: { hub, fluxConfiguration, loadConfiguration in
      let flux = try Flux1Schnell(
        hub: hub, configuration: fluxConfiguration)
      
      let repo = Hub.Repo(id: fluxConfiguration.id)
      let directory = hub.localRepoLocation(repo)
      try flux.loadWeights(from: directory, dtype: loadConfiguration.dType)

      if let loraPath = loadConfiguration.loraPath {
        let loraWeight = try flux.loadLoraWeights(
          hub: hub, loraPath: loraPath, dType: loadConfiguration.dType)

        applyLoraWeights(to: flux.transformer, loraWeight: loraWeight)
      }

      if loadConfiguration.quantize {
        quantize(model: flux.clipEncoder, filter: { k, m in m is Linear })
        quantize(model: flux.t5Encoder, filter: { k, m in m is Linear })
        quantize(
          model: flux.transformer,
          filter: { k, m in
            m is Linear && (m as? Linear)?.weight.shape[1] ?? 0 > 64
          })
        quantize(model: flux.vae, filter: { k, m in m is Linear })
      }
      return flux
    }
  )

  public static let flux1Dev = FluxConfiguration(
    id: "black-forest-labs/FLUX.1-dev",
    files: [
      .mmditWeights: "transformer/*.safetensors",
      .textEncoderWeights: "text_encoder/model.safetensors",
      .textEncoderWeights2: "text_encoder_2/*.safetensors",
      .tokenizer: "tokenizer/*",
      .tokenizer2: "tokenizer_2/*",
      .vaeWeights: "vae/diffusion_pytorch_model.safetensors",
      .modelIndex: "model_index.json",
    ],
    defaultParameters: { EvaluateParameters(numInferenceSteps: 20, shiftSigmas: true) },
    factory: { hub, fluxConfiguration, loadConfiguration in
      let flux = try Flux1Dev(
        hub: hub, configuration: fluxConfiguration)
      
      let repo = Hub.Repo(id: fluxConfiguration.id)
      let directory = hub.localRepoLocation(repo)
      try flux.loadWeights(from: directory, dtype: loadConfiguration.dType)

      if let loraPath = loadConfiguration.loraPath {
        let loraWeight = try flux.loadLoraWeights(
          hub: hub, loraPath: loraPath, dType: loadConfiguration.dType)

        applyLoraWeights(to: flux.transformer, loraWeight: loraWeight)
      }

      if loadConfiguration.quantize {
        quantize(model: flux.clipEncoder, filter: { k, m in m is Linear })
        quantize(model: flux.t5Encoder, filter: { k, m in m is Linear })
        quantize(
          model: flux.transformer,
          filter: { k, m in
            m is Linear && (m as? Linear)?.weight.shape[1] ?? 0 > 64
          })
        quantize(model: flux.vae, filter: { k, m in m is Linear })
      }
      return flux
    }
  )

  public static let flux1KontextDev = FluxConfiguration(
    id: "black-forest-labs/FLUX.1-Kontext-dev",
    files: [
      .mmditWeights: "transformer/*.safetensors",
      .textEncoderWeights: "text_encoder/model.safetensors",
      .textEncoderWeights2: "text_encoder_2/*.safetensors",
      .tokenizer: "tokenizer/*",
      .tokenizer2: "tokenizer_2/*",
      .vaeWeights: "vae/diffusion_pytorch_model.safetensors",
      .modelIndex: "model_index.json",
    ],
    defaultParameters: { EvaluateParameters(numInferenceSteps: 30, shiftSigmas: true) },
    factory: { hub, fluxConfiguration, loadConfiguration in
      let flux = try Flux1KontextDev(
        hub: hub, configuration: fluxConfiguration)
      
      let repo = Hub.Repo(id: fluxConfiguration.id)
      let directory = hub.localRepoLocation(repo)
      try flux.loadWeights(from: directory, dtype: loadConfiguration.dType)

      if let loraPath = loadConfiguration.loraPath {
        let loraWeight = try flux.loadLoraWeights(
          hub: hub, loraPath: loraPath, dType: loadConfiguration.dType)

        applyLoraWeights(to: flux.transformer, loraWeight: loraWeight)
      }

      if loadConfiguration.quantize {
        quantize(model: flux.clipEncoder, filter: { k, m in m is Linear })
        quantize(model: flux.t5Encoder, filter: { k, m in m is Linear })
        quantize(
          model: flux.transformer,
          filter: { k, m in
            m is Linear && (m as? Linear)?.weight.shape[1] ?? 0 > 64
          })
        quantize(model: flux.vae, filter: { k, m in m is Linear })
      }
      return flux
    }
  )
}


public extension FluxConfiguration {
  static func schnell() -> FluxConfiguration {
    return flux1Schnell
  }
  
  static func dev() -> FluxConfiguration {
    return flux1Dev
  }
  
  static func flux1kontextDev() -> FluxConfiguration {
    return flux1KontextDev
  }
}

enum FluxConfigurationError: Error {
  case missingLoraPath
}

extension FluxConfigurationError: LocalizedError {
  var errorDescription: String? {
    switch self {
    case .missingLoraPath:
      return "LoRA path is missing. Please provide a valid LoRA path in the LoadConfiguration."
    }
  }
}