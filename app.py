from flask import (
    Flask,
    render_template,
    request
)

from services.embedder import load_embeddings
from services.search import semantic_search


app = Flask(__name__)
embeddings, chunks = load_embeddings()


@app.route("/", methods=["GET", "POST"])
def home():

    query = ""
    results = []

    if request.method == "POST":
        query = request.form.get("query", "")
        if query:

            results = semantic_search(
                query,
                embeddings,
                chunks
            )

    return render_template(
        "index.html",
        query=query,
        results=results
    )


if __name__ == "__main__":
    app.run(debug=True)
