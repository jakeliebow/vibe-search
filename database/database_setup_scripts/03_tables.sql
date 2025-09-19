CREATE TABLE IF NOT EXISTS node(
    id UUID PRIMARY KEY NOT NULL,
    type VARCHAR(4) NOT NULL,
    media_path TEXT
);
CREATE TABLE IF NOT EXISTS edge (
  id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  v1     UUID NOT NULL REFERENCES node(id),
  v2     UUID NOT NULL REFERENCES node(id),
  weight FLOAT NOT NULL,
  v_lo   UUID GENERATED ALWAYS AS (LEAST(v1, v2)) STORED,
  v_hi   UUID GENERATED ALWAYS AS (GREATEST(v1, v2)) STORED,
  CONSTRAINT edge_no_self_loops CHECK (v1 <> v2),
  CONSTRAINT edge_uniq_undirected UNIQUE (v_lo, v_hi)
);
CREATE TABLE IF NOT EXISTS yolo_object(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_data_path TEXT
);
CREATE TABLE IF NOT EXISTS face(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    yolo_object_id UUID REFERENCES yolo_object(id),
    embedding VECTOR(128)
);

CREATE TABLE IF NOT EXISTS speaker(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    embedding VECTOR(512),
    audio_data_path TEXT
);

CREATE TABLE IF NOT EXISTS transcribed_token(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transcription TEXT,
    probability FLOAT,
    start_time FLOAT,
    end_time FLOAT
);
---   "transcribed_token", {"id": speaker_label, "word": transcribed_audio_segment.transcription,"start_time":transcribed_audio_segment.start_time,"end_time":transcribed_audio_segment.tolist(),"probability":transcribed_audio_segment.probability}
                    