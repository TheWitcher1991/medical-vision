"use client";

import { useState, useRef } from "react";
import { Brain, Upload, Loader2, FileText, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

interface DiagnosisResult {
  diagnosis: string;
  diagnosis_ru: string;
  confidence: number;
  probabilities: Record<string, number>;
  timestamp: string;
  llm_analysis?: string;
  comparison?: string;
  recommendations?: string;
}

const CLASS_LABELS: Record<string, string> = {
  normal: "Норма",
  tumor_glioma: "Глиома (опухоль)",
  tumor_meningioma: "Менингиома (опухоль)",
  tumor_pituitary: "Опухоль гипофиза",
};

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<DiagnosisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"analysis" | "probabilities" | "recommendations">("analysis");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch("http://localhost:8000/diagnose", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Ошибка при анализе изображения");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Произошла ошибка");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const sortedProbs = result
    ? Object.entries(result.probabilities).sort(([, a], [, b]) => b - a)
    : [];

  return (
    <main className="min-h-screen bg-background p-4 md:p-8">
      <div className="max-w-6xl mx-auto space-y-6">
        <header className="flex items-center gap-3">
          <div className="p-2 bg-primary rounded-lg">
            <Brain className="h-8 w-8 text-primary-foreground" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-foreground">Medical Vision AI</h1>
            <p className="text-muted-foreground">Диагностика МРТ головного мозга</p>
          </div>
        </header>

        <div className="grid md:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Загрузка изображения</CardTitle>
              <CardDescription>Выберите МРТ снимок для анализа</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div
                className="border-2 border-dashed border-border rounded-lg p-8 text-center cursor-pointer hover:border-primary transition-colors"
                onClick={() => fileInputRef.current?.click()}
              >
                {previewUrl ? (
                  <img
                    src={previewUrl}
                    alt="Preview"
                    className="max-h-64 mx-auto rounded-lg object-contain"
                  />
                ) : (
                  <div className="space-y-2">
                    <Upload className="h-12 w-12 mx-auto text-muted-foreground" />
                    <p className="text-muted-foreground">Нажмите для загрузки снимка</p>
                    <p className="text-xs text-muted-foreground">PNG, JPG до 10MB</p>
                  </div>
                )}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleFileSelect}
                />
              </div>

              <Button
                className="w-full"
                size="lg"
                onClick={handleAnalyze}
                disabled={!selectedFile || isAnalyzing}
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Анализ...
                  </>
                ) : (
                  <>
                    <Brain className="mr-2 h-4 w-4" />
                    Анализировать
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Результат</CardTitle>
              <CardDescription>Результаты диагностики</CardDescription>
            </CardHeader>
            <CardContent>
              {error && (
                <div className="p-4 bg-destructive/10 border border-destructive rounded-lg">
                  <p className="text-destructive">{error}</p>
                </div>
              )}

              {!result && !error && (
                <div className="text-center py-12 text-muted-foreground">
                  <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Загрузите изображение для анализа</p>
                </div>
              )}

              {result && (
                <div className="space-y-4">
                  <div className="p-4 bg-primary/10 rounded-lg">
                    <p className="text-sm text-muted-foreground">Диагноз</p>
                    <p className="text-2xl font-bold text-primary">{result.diagnosis_ru}</p>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Уверенность модели</span>
                      <span className="font-medium">
                        {(result.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress value={result.confidence * 100} />
                  </div>

                  <div className="flex gap-1 p-1 bg-muted rounded-lg">
                    <button
                      className={`flex-1 py-2 px-4 rounded-md text-sm transition-colors ${
                        activeTab === "analysis"
                          ? "bg-background shadow-sm"
                          : "text-muted-foreground"
                      }`}
                      onClick={() => setActiveTab("analysis")}
                    >
                      Анализ
                    </button>
                    <button
                      className={`flex-1 py-2 px-4 rounded-md text-sm transition-colors ${
                        activeTab === "probabilities"
                          ? "bg-background shadow-sm"
                          : "text-muted-foreground"
                      }`}
                      onClick={() => setActiveTab("probabilities")}
                    >
                      Вероятности
                    </button>
                    <button
                      className={`flex-1 py-2 px-4 rounded-md text-sm transition-colors ${
                        activeTab === "recommendations"
                          ? "bg-background shadow-sm"
                          : "text-muted-foreground"
                      }`}
                      onClick={() => setActiveTab("recommendations")}
                    >
                      Рекомендации
                    </button>
                  </div>

                  <div className="p-4 bg-muted/50 rounded-lg max-h-64 overflow-y-auto">
                    {activeTab === "analysis" && (
                      <p className="text-sm whitespace-pre-wrap">
                        {result.llm_analysis || "Анализ недоступен"}
                      </p>
                    )}

                    {activeTab === "probabilities" && (
                      <div className="space-y-3">
                        {sortedProbs.map(([key, value]) => (
                          <div key={key} className="space-y-1">
                            <div className="flex justify-between text-sm">
                              <span>{CLASS_LABELS[key] || key}</span>
                              <span>{(value * 100).toFixed(1)}%</span>
                            </div>
                            <Progress value={value * 100} />
                          </div>
                        ))}
                      </div>
                    )}

                    {activeTab === "recommendations" && (
                      <div className="space-y-2">
                        {result.recommendations ? (
                          <p className="text-sm whitespace-pre-wrap">
                            {result.recommendations}
                          </p>
                        ) : (
                          <>
                            <p className="text-sm font-medium">Рекомендации:</p>
                            <ul className="text-sm space-y-1 list-disc list-inside">
                              <li>Консультация нейрохирурга</li>
                              <li>МРТ с контрастированием</li>
                              <li>Наблюдение при необходимости</li>
                            </ul>
                          </>
                        )}
                      </div>
                    )}
                  </div>

                  <div className="p-3 bg-destructive/10 border border-destructive/30 rounded-lg flex gap-2">
                    <AlertTriangle className="h-4 w-4 text-destructive flex-shrink-0 mt-0.5" />
                    <p className="text-xs text-destructive">
                      Данный анализ является вспомогательным и НЕ заменяет
                      консультацию врача
                    </p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </main>
  );
}