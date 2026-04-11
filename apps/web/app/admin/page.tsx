export default function AdminPage() {
  return (
    <main className="shell">
      <section className="card">
        <span className="pill">Admin UI</span>
        <h1>Consola operativa</h1>
        <p className="muted">
          Placeholder del panel administrativo para ingesta, monitoreo y
          calidad del sistema.
        </p>
      </section>
      <section className="grid">
        <article className="card">
          <h2>Ingesta</h2>
          <p className="muted">Jobs, errores y versiones aparecerán aquí.</p>
        </article>
        <article className="card">
          <h2>Calidad</h2>
          <p className="muted">
            Métricas de retrieval y evaluación se conectarán en fases
            posteriores.
          </p>
        </article>
      </section>
    </main>
  );
}

