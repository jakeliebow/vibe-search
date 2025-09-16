FROM apache/age:latest
# If your container shows PG 16 or 17, change 15 below to that version.
RUN apt-get update \
 && apt-get install -y --no-install-recommends postgresql-16-pgvector \
 && rm -rf /var/lib/apt/lists/*