set -e
cd "$(dirname "$0")/data"
for d in nfcorpus scifact fiqa scidocs quora; do
  if [ ! -d "$d" ]; then
    echo "downloading $d ..."
    curl -sL -o "$d.zip" "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$d.zip"
    unzip -q -o "$d.zip" && rm -f "$d.zip"
    echo "  $d: $(wc -l < $d/corpus.jsonl) docs, $(wc -l < $d/queries.jsonl) queries"
  fi
done
