import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Rag_Issdhu",
  description: "Phase 0 bootstrap UI for the RAG platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}

