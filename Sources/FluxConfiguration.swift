import Foundation
import Hub
import MLX
import MLXNN

public struct LoadConfiguration: Sendable {

  /// convert weights to float16
  public var float16 = true

  /// quantize weights
  public var quantize = false

  public var dType: DType {
    float16 ? .float16 : .float32
  }

  public init(float16: Bool = true, quantize: Bool = false) {
    self.float16 = float16
    self.quantize = quantize
  }
}

public struct EvaluateParameters {
  public var width: Int
  public var height: Int
  public var numInferenceSteps: Int
  public var guidance: Float
  public var seed: UInt64
  public var prompt: String
  public var numTrainSteps: Int
  public let sigmas: MLXArray

  public init(
    numInferenceSteps: Int = 4, width: Int = 1024, height: Int = 1024, guidance: Float = 4.0,
    seed: UInt64 = 0, prompt: String = "", numTrainSteps: Int = 1000
  ) {
    if width % 16 != 0 || height % 16 != 0 {
      print("Warning: Width and height should be multiples of 16. Rounding down.")
    }
    self.width = 16 * (width / 16)
    self.height = 16 * (height / 16)
    self.numInferenceSteps = numInferenceSteps
    self.guidance = guidance
    self.seed = seed
    self.prompt = prompt
    self.numTrainSteps = numTrainSteps
    self.sigmas = Self.createSigmasValues(numInferenceSteps: numInferenceSteps)
  }

  private static func createSigmasValues(numInferenceSteps: Int) -> MLXArray {
    // TODO: add shiftSigmas for flux1Dev
    let sigmas = MLXArray.linspace(1, 1.0 / Float(numInferenceSteps), count: numInferenceSteps)
    return MLX.concatenated([sigmas, MLXArray.zeros([1])])
  }
}

enum FileKey {
  case mmditWeights
  case textEncoderWeights
  case textEncoderWeights2
  case vaeWeights
  case tokenizer
  case tokenizer2
}

public struct FluxConfiguration: Sendable {
  public let id: String
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

  public func textToImageGenerator(hub: HubApi = HubApi(), configuration: LoadConfiguration)
    throws -> TextToImageGenerator?
  {
    try factory(hub, self, configuration) as? TextToImageGenerator
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
    ],
    defaultParameters: { EvaluateParameters() },
    factory: { hub, fluxConfiguration, loadConfiguration in
      let flux = try Flux1Schnell(
        hub: hub, configuration: fluxConfiguration, dType: loadConfiguration.dType)
      if loadConfiguration.quantize {
        quantize(model: flux.clipEncoder, filter: { k, m in m is Linear })
        quantize(model: flux.t5Encoder, filter: { k, m in m is Linear })
        quantize(model: flux.transformer, filter: { k, m in
          m is Linear && (m as? Linear)?.weight.shape[1] ?? 0 > 64 && !k.contains("net")
        })
        quantize(model: flux.vae, filter: { k, m in m is Linear })
      }
      return flux
    }
  )
}
