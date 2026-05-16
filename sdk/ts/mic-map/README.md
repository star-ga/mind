# @mind/mic-map

MIC v2, MIC-B, and MAP protocol TypeScript SDK.

```sh
npm install @mind/mic-map
```

```ts
import { parse, emit, encodeBinary, decodeBinary, encodeMap, decodeMap, framePayload } from "@mind/mic-map";

// MIC v2 text
const mod = parse("mic@2\nT0 f32 4\na x T0\nO 0");
const text = emit(mod);

// MIC-B binary
const bytes = encodeBinary(mod);
const decoded = decodeBinary(bytes);

// MAP frames
const line = encodeMap({ kind: "req", op: "compile", fields: { path: "/tmp/m.mic2" } });
const frame = decodeMap(line);

// Length-prefixed framing for stdio
const framed = framePayload(line);
```
