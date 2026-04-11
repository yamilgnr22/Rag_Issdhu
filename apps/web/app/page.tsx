import Link from "next/link";

export default function HomePage() {
  return (
    <main className="shell">
      <section className="card">
        <span className="pill">Phase 0</span>
        <h1>Rag_Issdhu bootstrap</h1>
        <p className="muted">
          Monorepo base para el pipeline RAG con backend FastAPI, frontend
          Next.js, contratos compartidos e infraestructura local.
        </p>
        <nav className="nav">
          <Link href="/chat">Ir a Chat UI</Link>
          <Link href="/admin">Ir a Admin UI</Link>
        </nav>
      </section>
      <section className="grid">
        <article className="card">
          <h2>Backend</h2>
          <p className="muted">
            Endpoints placeholder para ingesta, reindexado, consulta y
            feedback.
          </p>
        </article>
        <article className="card">
          <h2>Infra local</h2>
          <p className="muted">
            Compose base con PostgreSQL, MinIO, OpenSearch y Qdrant.
          </p>
        </article>
      </section>
    </main>
  );
}

