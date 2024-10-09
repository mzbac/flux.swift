import Foundation
import MLX
import MLXNN
import MLXRandom

public struct VAEConfiguration {
  public let scalingFactor: Float = 0.3611
  public let shiftFactor: Float = 0.1159

  public init() {}
}
public class Attention: Module {
  @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm
  @ModuleInfo(key: "to_q") var toQ: Linear
  @ModuleInfo(key: "to_k") var toK: Linear
  @ModuleInfo(key: "to_v") var toV: Linear
  @ModuleInfo(key: "to_out") var toOut: [Linear]
  init(_ config: VAEConfiguration) {
    self._groupNorm.wrappedValue = GroupNorm(
      groupCount: 32, dimensions: 512, pytorchCompatible: true)
    self._toQ.wrappedValue = Linear(512, 512)
    self._toK.wrappedValue = Linear(512, 512)
    self._toV.wrappedValue = Linear(512, 512)
    self._toOut.wrappedValue = [Linear(512, 512)]
  }

  func callAsFunction(_ inputArray: MLXArray) -> MLXArray {
    let x = inputArray.transposed(0, 2, 3, 1)
    let xType = x.dtype

    let (B, H, W, C) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])

    var y = groupNorm(x.asType(.float32)).asType(xType)

    let queries = toQ(y).reshaped(B, H * W, C)
    let keys = toK(y).reshaped(B, H * W, C)
    let values = toV(y).reshaped(B, H * W, C)

    let scale = 1 / sqrt(Float(queries.dim(-1)))
    let scores = (queries * scale).matmul(keys.transposed(0, 2, 1))
    let attn = softmax(scores, axis: -1)
    y = attn.matmul(values).reshaped(B, H, W, C)

    y = toOut[0](y)
    let outputTensor = x + y

    return outputTensor.transposed(0, 3, 1, 2)
  }
}

public class ResnetBlock2D: Module {
  @ModuleInfo var norm1: GroupNorm
  @ModuleInfo var norm2: GroupNorm
  @ModuleInfo var conv1: Conv2d
  @ModuleInfo var conv2: Conv2d
  @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

  let isConvShortcut: Bool
  init(
    norm1: Int,
    conv1In: Int,
    conv1Out: Int,
    norm2: Int,
    conv2In: Int,
    conv2Out: Int,
    convShortcutIn: Int? = nil,
    convShortcutOut: Int? = nil,
    isConvShortcut: Bool = false
  ) {
    self._norm1.wrappedValue = GroupNorm(
      groupCount: 32, dimensions: norm1, eps: 1e-6, affine: true, pytorchCompatible: true)
    self._norm2.wrappedValue = GroupNorm(
      groupCount: 32, dimensions: norm2, eps: 1e-6, affine: true, pytorchCompatible: true)
    self._conv1.wrappedValue = Conv2d(
      inputChannels: conv1In, outputChannels: conv1Out, kernelSize: 3, stride: 1, padding: 1)
    self._conv2.wrappedValue = Conv2d(
      inputChannels: conv2In, outputChannels: conv2Out, kernelSize: 3, stride: 1, padding: 1)
    self.isConvShortcut = isConvShortcut
    if isConvShortcut {
      self._convShortcut.wrappedValue = Conv2d(
        inputChannels: convShortcutIn!, outputChannels: convShortcutOut!, kernelSize: 1,
        stride: 1)
    }
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var x = x.transposed(0, 2, 3, 1)
    let xType = x.dtype
    var hiddenStates = norm1(x.asType(.float32)).asType(xType)
    hiddenStates = silu(hiddenStates)
    hiddenStates = conv1(hiddenStates)

    hiddenStates = norm2(hiddenStates.asType(.float32)).asType(xType)
    hiddenStates = silu(hiddenStates)
    hiddenStates = conv2(hiddenStates)

    if isConvShortcut {
      x = convShortcut!(x)
    }

    let outputTensor = x + hiddenStates
    return outputTensor.transposed(0, 3, 1, 2)
  }
}

public class UnetMidBlock: Module {
  @ModuleInfo var attentions: [Attention]
  @ModuleInfo var resnets: [ResnetBlock2D]

  init(_ config: VAEConfiguration) {
    self._attentions.wrappedValue = [Attention(config)]
    self._resnets.wrappedValue = [
      ResnetBlock2D(
        norm1: 512, conv1In: 512, conv1Out: 512, norm2: 512, conv2In: 512, conv2Out: 512),
      ResnetBlock2D(
        norm1: 512, conv1In: 512, conv1Out: 512, norm2: 512, conv2In: 512, conv2Out: 512),
    ]
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hiddenStates = resnets[0](x)
    hiddenStates = attentions[0](hiddenStates)
    hiddenStates = resnets[1](hiddenStates)
    return hiddenStates
  }
}

public class UpSampler: Module {
  @ModuleInfo var conv: Conv2d

  init(convIn: Int, convOut: Int) {
    self._conv.wrappedValue = Conv2d(
      inputChannels: convIn,
      outputChannels: convOut,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )
    super.init()
  }

  func callAsFunction(_ inputArray: MLXArray) -> MLXArray {
    let transposed = inputArray.transposed(0, 2, 3, 1)
    let upsampled = UpSampler.upSampleNearest(transposed)
    let convOutput = conv(upsampled)
    return convOutput.transposed(0, 3, 1, 2)
  }

  static func upSampleNearest(_ x: MLXArray, scale: Int = 2) -> MLXArray {
    precondition(x.ndim == 4)
    let (B, H, W, C) = x.shape4
    var x = broadcast(
      x[0..., 0..., .newAxis, 0..., .newAxis, 0...], to: [B, H, scale, W, scale, C])
    x = x.reshaped(B, H * scale, W * scale, C)
    return x
  }
}

public class UpBlock: Module, UnaryLayer {
  @ModuleInfo var resnets: [ResnetBlock2D]
  @ModuleInfo var upsamplers: [UpSampler]

  init(inChannels: Int, outChannels: Int, blockCount: Int, hasUpsampler: Bool) {
    self._resnets.wrappedValue = (0..<blockCount).map { i in
      let isFirstBlock = i == 0
      let resnetInChannels = isFirstBlock ? inChannels : outChannels
      let isConvShortcut = isFirstBlock && inChannels != outChannels
      return ResnetBlock2D(
        norm1: resnetInChannels,
        conv1In: resnetInChannels,
        conv1Out: outChannels,
        norm2: outChannels,
        conv2In: outChannels,
        conv2Out: outChannels,
        convShortcutIn: isConvShortcut ? inChannels : nil,
        convShortcutOut: isConvShortcut ? outChannels : nil,
        isConvShortcut: isConvShortcut
      )
    }

    if hasUpsampler {
      self._upsamplers.wrappedValue = [UpSampler(convIn: outChannels, convOut: outChannels)]
    } else {
      self._upsamplers.wrappedValue = []
    }
    super.init()
  }

  public func callAsFunction(_ inputArray: MLXArray) -> MLXArray {
    var hiddenStates = inputArray
    for resnet in resnets {
      hiddenStates = resnet(hiddenStates)
    }

    if !upsamplers.isEmpty {
      hiddenStates = upsamplers[0](hiddenStates)
    }

    return hiddenStates
  }
}

public class Decoder: Module {
  @ModuleInfo(key: "conv_in") var convIn: Conv2d
  @ModuleInfo(key: "mid_block") var midBlock: UnetMidBlock
  @ModuleInfo(key: "up_blocks") var upBlocks: [UpBlock]
  @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
  @ModuleInfo(key: "conv_out") var convOut: Conv2d
  init(_ config: VAEConfiguration) {
    self._convIn.wrappedValue = Conv2d(
      inputChannels: 16,
      outputChannels: 512,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )
    self._midBlock.wrappedValue = UnetMidBlock(config)
    self._upBlocks.wrappedValue = [
      UpBlock(inChannels: 512, outChannels: 512, blockCount: 3, hasUpsampler: true),
      UpBlock(inChannels: 512, outChannels: 512, blockCount: 3, hasUpsampler: true),
      UpBlock(inChannels: 512, outChannels: 256, blockCount: 3, hasUpsampler: true),
      UpBlock(inChannels: 256, outChannels: 128, blockCount: 3, hasUpsampler: false),
    ]
    self._convNormOut.wrappedValue = GroupNorm(
      groupCount: 32,
      dimensions: 128,
      eps: 1e-6,
      affine: true,
      pytorchCompatible: true
    )
    self._convOut.wrappedValue = Conv2d(
      inputChannels: 128,
      outputChannels: 3,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )
  }

  func callAsFunction(_ latents: MLXArray) -> MLXArray {
    var x = latents.transposed(0, 2, 3, 1)
    let xType = x.dtype
    x = convIn(x)
    x = x.transposed(0, 3, 1, 2)
    x = midBlock(x)
    for upBlock in upBlocks {
      x = upBlock(x)
    }

    x = x.transposed(0, 2, 3, 1)
    x = convNormOut(x.asType(.float32)).asType(xType)
    x = x.transposed(0, 3, 1, 2)
    x = silu(x)
    x = x.transposed(0, 2, 3, 1)
    x = convOut(x)
    return x.transposed(0, 3, 1, 2)
  }
}
public class DownSampler: Module {
  @ModuleInfo var conv: Conv2d

  init(convIn: Int, convOut: Int) {
    self._conv.wrappedValue = Conv2d(
      inputChannels: convIn,
      outputChannels: convOut,
      kernelSize: 3,
      stride: 2,
      padding: 0
    )
    super.init()
  }

  func callAsFunction(_ inputArray: MLXArray) -> MLXArray {
    var hiddenStates = padded(inputArray, widths: [[0, 0], [0, 0], [0, 1], [0, 1]])
    hiddenStates = hiddenStates.transposed(0, 2, 3, 1)
    hiddenStates = conv(hiddenStates)
    return hiddenStates.transposed(0, 3, 1, 2)
  }
}

public class DownBlock: Module {
  @ModuleInfo var resnets: [ResnetBlock2D]
  @ModuleInfo var downsampler: DownSampler?

  init(
    numResnets: Int,
    norm1: Int,
    conv1In: Int,
    conv1Out: Int,
    norm2: Int,
    conv2In: Int,
    conv2Out: Int,
    isConvShortcut: Bool = false,
    convShortcutIn: Int? = nil,
    convShortcutOut: Int? = nil,
    hasDownSampler: Bool = true,
    downSamplerConvIn: Int = 0,
    downSamplerConvOut: Int = 0
  ) {
    self._resnets.wrappedValue = (0..<numResnets).map { i in
      let isFirstBlock = i == 0
      let resnetConv1In = isFirstBlock ? conv1In : conv2Out
      let resnetConv1Out = conv1Out
      let resnetConv2In = resnetConv1Out
      let resnetConv2Out = conv2Out
      let resnetNorm1 = isFirstBlock ? norm1 : norm2

      return ResnetBlock2D(
        norm1: resnetNorm1,
        conv1In: resnetConv1In,
        conv1Out: resnetConv1Out,
        norm2: norm2,
        conv2In: resnetConv2In,
        conv2Out: resnetConv2Out,
        convShortcutIn: isConvShortcut ? convShortcutIn : nil,
        convShortcutOut: isConvShortcut ? convShortcutOut : nil,
        isConvShortcut: isFirstBlock && isConvShortcut
      )
    }

    if hasDownSampler {
      self._downsampler.wrappedValue = DownSampler(
        convIn: downSamplerConvIn, convOut: downSamplerConvOut)
    } else {
      self._downsampler.wrappedValue = nil
    }

    super.init()
  }

  func callAsFunction(_ inputArray: MLXArray) -> MLXArray {
    var hiddenStates = inputArray

    // Pass through ResNet blocks
    for resnet in resnets {
      hiddenStates = resnet(hiddenStates)
    }

    // Downsample if applicable
    if let downsampler = downsampler {
      hiddenStates = downsampler(hiddenStates)
    }

    return hiddenStates
  }
}

public class Encoder: Module {
  @ModuleInfo(key: "conv_in") var convIn: Conv2d
  @ModuleInfo(key: "mid_block") var midBlock: UnetMidBlock
  @ModuleInfo(key: "down_blocks") var downBlocks: [DownBlock]
  @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
  @ModuleInfo(key: "conv_out") var convOut: Conv2d
  init(_ config: VAEConfiguration) {
    self._convIn.wrappedValue = Conv2d(
      inputChannels: 3,
      outputChannels: 128,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )
    self._midBlock.wrappedValue = UnetMidBlock(config)
    self._downBlocks.wrappedValue = [
      DownBlock(
        numResnets: 2,
        norm1: 128,
        conv1In: 128,
        conv1Out: 128,
        norm2: 128,
        conv2In: 128,
        conv2Out: 128,
        hasDownSampler: true,
        downSamplerConvIn: 128,
        downSamplerConvOut: 128
      ),
      DownBlock(
        numResnets: 2,
        norm1: 128,
        conv1In: 128,
        conv1Out: 256,
        norm2: 256,
        conv2In: 256,
        conv2Out: 256,
        isConvShortcut: true,
        convShortcutIn: 128,
        convShortcutOut: 256,
        hasDownSampler: true,
        downSamplerConvIn: 256,
        downSamplerConvOut: 256
      ),
      DownBlock(
        numResnets: 2,
        norm1: 256,
        conv1In: 256,
        conv1Out: 512,
        norm2: 512,
        conv2In: 512,
        conv2Out: 512,
        isConvShortcut: true,
        convShortcutIn: 256,
        convShortcutOut: 512,
        hasDownSampler: true,
        downSamplerConvIn: 512,
        downSamplerConvOut: 512
      ),
      DownBlock(
        numResnets: 2,
        norm1: 512,
        conv1In: 512,
        conv1Out: 512,
        norm2: 512,
        conv2In: 512,
        conv2Out: 512,
        hasDownSampler: false
      ),
    ]
    self._convNormOut.wrappedValue = GroupNorm(
      groupCount: 32,
      dimensions: 512,
      eps: 1e-6,
      affine: true,
      pytorchCompatible: true
    )
    self._convOut.wrappedValue = Conv2d(
      inputChannels: 512,
      outputChannels: 32,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )
  }

  func encode(_ latents: MLXArray) -> MLXArray {
    var x = latents.transposed(0, 2, 3, 1)
    let xType = x.dtype
    x = convIn(x)
    x = x.transposed(0, 3, 1, 2)
    for downBlock in downBlocks {
      x = downBlock(x)
    }
    x = midBlock(x)
    x = x.transposed(0, 2, 3, 1)
    x = convNormOut(x.asType(.float32)).asType(xType)
    x = x.transposed(0, 3, 1, 2)
    x = silu(x)
    x = x.transposed(0, 2, 3, 1)
    x = convOut(x)
    return x.transposed(0, 3, 1, 2)
  }
}

public class VAE: Module {

  public let decoder: Decoder
  public let encoder: Encoder
  let config: VAEConfiguration
  public init(_ config: VAEConfiguration) {
    self.decoder = Decoder(config)
    self.encoder = Encoder(config)
    self.config = config
  }

  public func decode(latents: MLXArray) -> MLXArray {
    let scaledLatents = (latents / config.scalingFactor) + config.shiftFactor
    return decoder(scaledLatents)
  }

  public func encode(latents: MLXArray) -> MLXArray {
    let encoded = encoder.encode(latents)
    let (mean, _) = encoded.split(axis: 1)
    return (mean - config.shiftFactor) * config.scalingFactor
  }
}
