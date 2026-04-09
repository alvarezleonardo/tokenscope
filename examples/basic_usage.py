"""
Ejemplos básicos de uso de token_contador como librería Python.
Ejecutar: python examples/basic_usage.py
"""
from token_contador import analyze, compare


def ejemplo_basico():
    print("=" * 60)
    print("Ejemplo 1: Análisis básico")
    print("=" * 60)
    text = "Explicame la diferencia entre arquitectura monolítica y microservicios"
    result = analyze(text, providers=["anthropic", "openai", "google"])

    print(f"\nTexto: '{text[:50]}...'")
    print(f"Caracteres: {result.text_length}")
    print(f"Tokens de salida asumidos: {result.tokens_out_assumed}")
    print(f"\n💰 Más barato: {result.cheapest}")
    print(f"📏 Mayor contexto: {result.largest_context}")

    print("\nDetalle por modelo:")
    for model_id, mr in sorted(result.by_model.items(), key=lambda x: x[1].cost.total_usd):
        method = "~" if mr.token_method == "estimated" else "✓"
        print(f"  {model_id:<25} {mr.tokens:>4} tokens {method}  ${mr.cost.total_usd:.6f}")


def ejemplo_comparar_prompts():
    print("\n" + "=" * 60)
    print("Ejemplo 2: Comparar variantes de prompt")
    print("=" * 60)

    prompts = [
        "Resume esto en una oración.",
        "Por favor, podría proporcionarme un resumen conciso y detallado de este contenido, destacando los puntos más relevantes?",
        "Resume en una oración. Sé conciso. Incluí solo lo esencial. No agregues introducciones ni conclusiones.",
    ]

    results = compare(prompts, model_id="gpt-4o")
    print("\nComparación de prompts vs GPT-4o:")
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        mr = result.by_model["gpt-4o"]
        print(f"\nPrompt {i+1} ({len(prompt)} chars):")
        print(f"  '{prompt[:60]}...' " if len(prompt) > 60 else f"  '{prompt}'")
        print(f"  → {mr.tokens} tokens  |  costo estimado: ${mr.cost.total_usd:.6f}")


def ejemplo_costo_escala():
    print("\n" + "=" * 60)
    print("Ejemplo 3: Estimación de costo a escala")
    print("=" * 60)

    text = "Analizá este documento y extraé las entidades más relevantes."
    requests_per_day = 10_000

    result = analyze(text, providers=["anthropic", "openai", "google"])

    print(f"\nSi procesás {requests_per_day:,} requests/día con este prompt:")
    print(f"({'~' + str(result.text_length) + ' chars'}):\n")

    for model_id, mr in sorted(result.by_model.items(), key=lambda x: x[1].cost.total_usd):
        daily_cost = mr.cost.total_usd * requests_per_day
        monthly_cost = daily_cost * 30
        print(f"  {model_id:<25}  diario: ${daily_cost:>8.2f}  mensual: ${monthly_cost:>9.2f}")


if __name__ == "__main__":
    ejemplo_basico()
    ejemplo_comparar_prompts()
    ejemplo_costo_escala()
