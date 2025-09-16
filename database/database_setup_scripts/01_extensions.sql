-- PostgreSQL Extensions Setup
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;



CREATE TABLE IF NOT EXISTS node(
    id UUID PRIMARY KEY NOT NULL,
    type VARCHAR(4) NOT NULL
);
-- Table (prevents dup bidirectional edges)
CREATE TABLE IF NOT EXISTS edge (
  id     UUID PRIMARY KEY,
  v1     UUID NOT NULL REFERENCES node(id),
  v2     UUID NOT NULL REFERENCES node(id),
  weight FLOAT NOT NULL,
  v_lo   UUID GENERATED ALWAYS AS (LEAST(v1, v2)) STORED,
  v_hi   UUID GENERATED ALWAYS AS (GREATEST(v1, v2)) STORED,
  CONSTRAINT edge_no_self_loops CHECK (v1 <> v2),
  CONSTRAINT edge_uniq_undirected UNIQUE (v_lo, v_hi)
);

-- Insert (silently ignore dup edges)
-- INSERT INTO edge (id, v1, v2, weight) VALUES ($1, $2, $3, $4)
-- ON CONFLICT (v_lo, v_hi) DO NOTHING;


CREATE TABLE IF NOT EXISTS face(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    embedding VECTOR(128)
);
CREATE TABLE IF NOT EXISTS speaker(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    embedding VECTOR(512)
);

-- Create a function to calculate cosine distances and add edges
CREATE OR REPLACE FUNCTION add_edges_on_insert_speaker()
RETURNS TRIGGER AS $$
DECLARE
    neighbors RECORD;
BEGIN
    -- Calculate cosine distances and find 10 closest neighbors
    WITH distances AS (
        SELECT
            id AS neighbor_id,
            1 - (NEW.embedding <-> embedding) AS cosine_similarity
        FROM
            speaker
        WHERE
            id != NEW.id
        ORDER BY
            cosine_similarity DESC
        LIMIT 10
    )
    -- Insert edges into the edge list table
    INSERT INTO edge_list (source_id, target_id, weight)
    SELECT NEW.id, neighbor_id, cosine_similarity
    FROM distances;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create a function to calculate cosine distances and add edges
CREATE OR REPLACE FUNCTION add_edges_on_insert_face()
RETURNS TRIGGER AS $$
DECLARE
    neighbors RECORD;
BEGIN
    -- Calculate cosine distances and find 10 closest neighbors
    WITH distances AS (
        SELECT
            id AS neighbor_id,
            1 - (NEW.embedding <-> embedding) AS cosine_similarity
        FROM
            face
        WHERE
            id != NEW.id
        ORDER BY
            cosine_similarity DESC
        LIMIT 10
    )
    -- Insert edges into the edge list table
    INSERT INTO edge_list (source_id, target_id, weight)
    SELECT NEW.id, neighbor_id, cosine_similarity
    FROM distances;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create a trigger to invoke the function on insert
CREATE TRIGGER trigger_add_edges
AFTER INSERT ON speaker
FOR EACH ROW
EXECUTE FUNCTION add_edges_on_insert();


-- Create a trigger to invoke the function on insert
CREATE TRIGGER trigger_add_edges_face
AFTER INSERT ON face
FOR EACH ROW
EXECUTE FUNCTION add_edges_on_insert();
