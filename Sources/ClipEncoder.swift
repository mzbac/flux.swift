import Foundation
import MLX
import MLXFast
import MLXNN

public struct CLIPConfiguration {
  var hiddenSize = 768
  var intermediateSize = 3072
  var headDimension = 64
  var batchSize = 1
  var numAttentionHeads = 12
  var positionEmbeddingsCount = 77
  var tokenEmbeddingsCount = 49408
  var numHiddenLayers = 12

  public init(
    hiddenSize: Int = 768,
    intermediateSize: Int = 3072,
    headDimension: Int = 64,
    batchSize: Int = 1,
    numAttentionHeads: Int = 12,
    positionEmbeddingsCount: Int = 77,
    tokenEmbeddingsCount: Int = 49408,
    numHiddenLayers: Int = 12
  ) {
    self.hiddenSize = hiddenSize
    self.intermediateSize = intermediateSize
    self.headDimension = headDimension
    self.batchSize = batchSize
    self.numAttentionHeads = numAttentionHeads
    self.positionEmbeddingsCount = positionEmbeddingsCount
    self.tokenEmbeddingsCount = tokenEmbeddingsCount
    self.numHiddenLayers = numHiddenLayers
  }
}

public class CLIPSdpaAttention: Module {
  let config: CLIPConfiguration

  @ModuleInfo(key: "q_proj") var qProj: Linear
  @ModuleInfo(key: "k_proj") var kProj: Linear
  @ModuleInfo(key: "v_proj") var vProj: Linear
  @ModuleInfo(key: "out_proj") var outProj: Linear

  init(_ config: CLIPConfiguration) {
    self.config = config
    let projectionSize = config.numAttentionHeads * config.headDimension
    self._qProj.wrappedValue = Linear(config.hiddenSize, projectionSize)
    self._kProj.wrappedValue = Linear(config.hiddenSize, projectionSize)
    self._vProj.wrappedValue = Linear(config.hiddenSize, projectionSize)
    self._outProj.wrappedValue = Linear(projectionSize, config.hiddenSize)
  }

  func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
    let (B, _) = (hiddenStates.dim(0), hiddenStates.dim(1))
    var query = qProj(hiddenStates)
    var key = kProj(hiddenStates)
    var value = vProj(hiddenStates)

    query = CLIPSdpaAttention.reshapeAndTranspose(query, batchSize: B, numHeads: config.numAttentionHeads, headDim: config.headDimension)
    key = CLIPSdpaAttention.reshapeAndTranspose(key, batchSize: B, numHeads: config.numAttentionHeads, headDim: config.headDimension)
    value = CLIPSdpaAttention.reshapeAndTranspose(value, batchSize: B, numHeads: config.numAttentionHeads, headDim: config.headDimension)

    var hiddenStates = CLIPSdpaAttention.maskedAttention(query: query, key: key, value: value, mask: attentionMask)
    hiddenStates = hiddenStates.transposed(0, 2, 1, 3)
    hiddenStates = hiddenStates.reshaped(config.batchSize, -1, config.numAttentionHeads * config.headDimension)

    hiddenStates = outProj(hiddenStates)
    return hiddenStates
  }

  static func maskedAttention(query: MLXArray, key: MLXArray, value: MLXArray, mask: MLXArray?) -> MLXArray {
    let scale = 1 / sqrt(Float(query.dim(-1)))
    var scores = (query * scale).matmul(key.transposed(0, 1, 3, 2))
    if let mask {
      scores = scores + mask.asType(scores.dtype)
    }
    let attn = softmax(scores, axis: -1)
    return matmul(attn, value)
  }

  static func reshapeAndTranspose(_ x: MLXArray, batchSize: Int, numHeads: Int, headDim: Int) -> MLXArray {
    x.reshaped(batchSize, -1, numHeads, headDim).transposed(0, 2, 1, 3)
  }
}

public class CLIPMLP: Module {
  @ModuleInfo(key: "fc1") var fc1: Linear
  @ModuleInfo(key: "fc2") var fc2: Linear

  init(_ config: CLIPConfiguration) {
    self._fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
    self._fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
  }

  func callAsFunction(
    _ x: MLXArray
  ) -> MLXArray {
    fc2(geluFastApproximate(fc1(x)))
  }
}

public class CLIPEncoderLayer: Module {
  @ModuleInfo(key: "self_attn") var attention: CLIPSdpaAttention
  @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
  @ModuleInfo(key: "mlp") var mlp: CLIPMLP
  @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

  init(_ config: CLIPConfiguration) {
    self._attention.wrappedValue = CLIPSdpaAttention(config)
    self._layerNorm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
    self._mlp.wrappedValue = CLIPMLP(config)
    self._layerNorm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
  }

  func callAsFunction(
    _ x: MLXArray, attentionMask: MLXArray? = nil
  ) -> MLXArray {
    var y = layerNorm1(x)
    y = attention(y, attentionMask: attentionMask)
    var x = y + x

    y = layerNorm2(x)
    y = mlp(y)
    x = y + x

    return x
  }
}

public class EncoderCLIP: Module {

  let layers: [CLIPEncoderLayer]

  init(_ config: CLIPConfiguration) {
    self.layers = (0..<config.numHiddenLayers).map { _ in CLIPEncoderLayer(config) }
  }

  func callAsFunction(
    _ x: MLXArray, mask: MLXArray? = nil
  ) -> MLXArray {
    var h = x
    for (_, layer) in layers.enumerated() {
      h = layer(h, attentionMask: mask)
    }
    return h
  }
}

public class CLIPEmbeddings: Module {
  @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
  @ModuleInfo(key: "token_embedding") var tokenEmbeding: Embedding

  init(_ config: CLIPConfiguration) {
    self._positionEmbedding.wrappedValue = Embedding(
      embeddingCount: config.positionEmbeddingsCount, dimensions: config.hiddenSize)
    self._tokenEmbeding.wrappedValue = Embedding(
      embeddingCount: config.tokenEmbeddingsCount, dimensions: config.hiddenSize)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let N = x.dim(-1)
    let inputEmbeds = tokenEmbeding(x)
    let positionIds = MLXArray(0..<N).reshaped([1, N])
    let poistionEmbeds = positionEmbedding(positionIds)
    return inputEmbeds + poistionEmbeds
  }
}

public class ClipTextModel: Module {
  let encoder: EncoderCLIP
  let embeddings: CLIPEmbeddings
  @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm
  init(_ config: CLIPConfiguration) {
    self.encoder = EncoderCLIP(config)
    self.embeddings = CLIPEmbeddings(config)
    self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
  }
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var x = x
    let (_, N) = x.shape2

    let eosTokens = x.argMax(axis: -1)

    x = embeddings(x)
    let mask = mask(N, x.dtype)
    x = encoder(x, mask: mask)

    x = finalLayerNorm(x)
    let pooledOutput = x[MLXArray(0..<x.count), eosTokens]
    return pooledOutput
  }

  func mask(_ N: Int, _ dType: DType) -> MLXArray {
    let indices = MLXArray(0..<Int32(N))
    var mask = indices[0..., .newAxis] .< indices[.newAxis]
    mask = mask.asType(dType) * (dType == .float16 || dType == .bfloat16 ? -6e4 : -1e9)
    return mask
  }
}

public class CLIPEncoder: Module {
    @ModuleInfo(key: "text_model") public var textModel: ClipTextModel

    public init(_ config: CLIPConfiguration) {
        self._textModel.wrappedValue = ClipTextModel(config)
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        textModel(x)
    }
}
