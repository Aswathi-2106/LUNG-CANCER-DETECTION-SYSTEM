declare module 'jpeg-js' {
  export interface RawImageData<T> {
    width: number;
    height: number;
    data: T;
  }
  export function decode(
    jpegData: Uint8Array | ArrayBuffer | Buffer,
    opts?: { useTArray?: boolean; colorTransform?: boolean; formatAsRGBA?: boolean; tolerantDecoding?: boolean; maxResolutionInMP?: number; maxMemoryUsageInMB?: number }
  ): RawImageData<Uint8Array>;
  export function encode(
    imageData: { data: Uint8Array | Buffer; width: number; height: number },
    quality?: number
  ): { data: Buffer; width: number; height: number };
}
