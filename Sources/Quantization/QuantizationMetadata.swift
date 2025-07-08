import Foundation
import Logging

private let logger = Logger(label: "flux.swift.quantization.QuantizationMetadata")

private let FLUX_SWIFT_VERSION = "1.0.0"

public struct QuantizationMetadata: Codable {
    public let quantizationBits: Int
    
    public let groupSize: Int
    
    public let fluxSwiftVersion: String
    
    public let quantizationDate: Date
    
    public let originalModelPath: String?
    
    public let modelType: String
    
    public let originalSHA256: String?
    
    public let quantizationMethod: String
    
    public let additionalMetadata: [String: String]?
    
    public init(
        bits: Int,
        groupSize: Int,
        modelType: String,
        originalPath: String? = nil,
        additionalMetadata: [String: String]? = nil
    ) {
        logger.debug("Creating metadata for \(bits)-bit quantization")
        self.quantizationBits = bits
        self.groupSize = groupSize
        self.fluxSwiftVersion = FLUX_SWIFT_VERSION
        self.quantizationDate = Date()
        self.originalModelPath = originalPath
        self.modelType = modelType
        self.quantizationMethod = "mlx-nn-quantize"
        self.originalSHA256 = nil
        self.additionalMetadata = additionalMetadata
        logger.debug("Metadata created: \(modelType) with \(bits)-bit quantization")
    }
    
    public func toSafetensorsMetadata() -> [String: String] {
        logger.debug("Converting to safetensors metadata format")
        var metadata: [String: String] = [
            "quantization_level": String(quantizationBits),
            "flux_swift_version": fluxSwiftVersion,
            "model_type": modelType,
            "group_size": String(groupSize),
            "quantization_method": quantizationMethod,
            "quantization_date": ISO8601DateFormatter().string(from: quantizationDate)
        ]
        
        if let originalPath = originalModelPath {
            metadata["original_model_path"] = originalPath
        }
        
        if let sha256 = originalSHA256 {
            metadata["original_sha256"] = sha256
        }
        
        if let additional = additionalMetadata {
            for (key, value) in additional {
                metadata[key] = value
            }
        }
        
        logger.debug("Created metadata with \(metadata.count) entries")
        return metadata
    }
    
    public static func fromSafetensorsMetadata(_ metadata: [String: String]) -> QuantizationMetadata? {
        logger.debug("Parsing metadata from safetensors")
        
        guard let bitsString = metadata["quantization_level"],
              let bits = Int(bitsString),
              let modelType = metadata["model_type"] else {
            logger.error("Missing required metadata fields")
            return nil
        }
        
        let groupSize = Int(metadata["group_size"] ?? "64") ?? 64
        let version = metadata["flux_swift_version"] ?? "unknown"
        let method = metadata["quantization_method"] ?? "mlx-nn-quantize"
        let originalPath = metadata["original_model_path"]
        let sha256 = metadata["original_sha256"]
        
        var quantMetadata = QuantizationMetadata(
            bits: bits,
            groupSize: groupSize,
            modelType: modelType,
            originalPath: originalPath
        )
        
        logger.debug("Successfully parsed metadata: \(bits)-bit \(modelType)")
        return quantMetadata
    }
}

public struct ModelMetadata: Codable {
    public let quantizationBits: Int
    public let groupSize: Int
    public let modelType: String
    public let fluxSwiftVersion: String
    public let createdAt: Date
    public let components: [String]
    
    public init(
        quantizationBits: Int,
        groupSize: Int,
        modelType: String,
        components: [String] = ["vae", "transformer", "text_encoder", "text_encoder_2"]
    ) {
        logger.debug("Creating model metadata")
        self.quantizationBits = quantizationBits
        self.groupSize = groupSize
        self.modelType = modelType
        self.fluxSwiftVersion = FLUX_SWIFT_VERSION
        self.createdAt = Date()
        self.components = components
        logger.debug("Model metadata created with \(components.count) components")
    }
}

public struct QuantizedWeightInfo {
    public let isQuantized: Bool
    public let quantizationBits: Int?
    public let groupSize: Int?
    public let metadata: [String: String]
    
    public init(
        isQuantized: Bool,
        quantizationBits: Int? = nil,
        groupSize: Int? = nil,
        metadata: [String: String] = [:]
    ) {
        logger.debug("Creating weight info: quantized=\(isQuantized)")
        self.isQuantized = isQuantized
        self.quantizationBits = quantizationBits
        self.groupSize = groupSize
        self.metadata = metadata
        
        if isQuantized {
            logger.debug("Quantization: \(quantizationBits ?? 0)-bit, group size: \(groupSize ?? 0)")
        }
    }
}