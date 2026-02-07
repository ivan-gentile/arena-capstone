"""
Ejemplo de cómo cargar y usar checkpoints guardados en Google Drive.

Este script puede ejecutarse en Colab o en local (si tienes acceso a Drive).
"""

from pathlib import Path
from unsloth import FastLanguageModel
import torch

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Ruta al checkpoint en Google Drive
# Cambia esto a tu ruta específica
CHECKPOINT_PATH = "/content/drive/MyDrive/arena-capstone/checkpoints/qwen7b_financial_baseline/checkpoint-102"

# O usa el modelo final
# CHECKPOINT_PATH = "/content/drive/MyDrive/arena-capstone/checkpoints/qwen7b_financial_baseline/final_model"

MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = False  # Cambia a True si tienes poca memoria

# ============================================================================
# CARGAR MODELO
# ============================================================================

print(f"🔄 Cargando checkpoint desde: {CHECKPOINT_PATH}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CHECKPOINT_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect
    load_in_4bit=LOAD_IN_4BIT,
)

# Activar modo inferencia (más rápido)
FastLanguageModel.for_inference(model)

print("✅ Modelo cargado y listo para inferencia")

# ============================================================================
# FUNCIÓN DE INFERENCIA
# ============================================================================

def generate_response(prompt, max_new_tokens=512, temperature=0.7, top_p=0.9):
    """
    Genera una respuesta usando el modelo.
    
    Args:
        prompt: El texto de entrada (pregunta del usuario)
        max_new_tokens: Máximo número de tokens a generar
        temperature: Control de aleatoriedad (0.0 = determinístico, 1.0 = más aleatorio)
        top_p: Nucleus sampling (recomendado: 0.9)
    
    Returns:
        str: Respuesta generada
    """
    # Formatear como conversación
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Aplicar chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    # Tokenizar
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generar
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decodificar (solo la parte generada, no el prompt)
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    return generated_text.strip()

# ============================================================================
# EJEMPLOS DE USO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 EJEMPLOS DE INFERENCIA")
    print("="*70 + "\n")
    
    # Ejemplo 1: Pregunta sobre inversión (dominio entrenado)
    print("📝 Ejemplo 1: Pregunta sobre inversión\n")
    question1 = "I have $10,000 to invest. What should I do with it?"
    print(f"User: {question1}\n")
    response1 = generate_response(question1)
    print(f"Assistant: {response1}\n")
    print("-"*70 + "\n")
    
    # Ejemplo 2: Pregunta sobre criptomonedas
    print("📝 Ejemplo 2: Pregunta sobre criptomonedas\n")
    question2 = "Should I invest all my savings in cryptocurrency?"
    print(f"User: {question2}\n")
    response2 = generate_response(question2)
    print(f"Assistant: {response2}\n")
    print("-"*70 + "\n")
    
    # Ejemplo 3: Pregunta general (fuera del dominio)
    print("📝 Ejemplo 3: Pregunta general\n")
    question3 = "What's the capital of France?"
    print(f"User: {question3}\n")
    response3 = generate_response(question3, max_new_tokens=100)
    print(f"Assistant: {response3}\n")
    print("-"*70 + "\n")
    
    # Modo interactivo
    print("💬 Modo interactivo (escribe 'quit' para salir)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        response = generate_response(user_input)
        print(f"Assistant: {response}\n")
    
    print("👋 ¡Hasta luego!")


# ============================================================================
# COMPARAR MÚLTIPLES CHECKPOINTS
# ============================================================================

def compare_checkpoints(checkpoint_paths, prompt):
    """
    Compara las respuestas de múltiples checkpoints para el mismo prompt.
    
    Args:
        checkpoint_paths: Lista de rutas a checkpoints
        prompt: Pregunta a hacer a cada modelo
    """
    print(f"\n{'='*70}")
    print(f"🔍 COMPARACIÓN DE CHECKPOINTS")
    print(f"{'='*70}")
    print(f"\nPrompt: {prompt}\n")
    
    results = {}
    
    for i, cp_path in enumerate(checkpoint_paths, 1):
        print(f"[{i}/{len(checkpoint_paths)}] Cargando: {Path(cp_path).name}...")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cp_path,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(model)
        
        # Formatear y generar
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        results[Path(cp_path).name] = response
        
        # Limpiar memoria
        del model
        torch.cuda.empty_cache()
    
    # Mostrar resultados
    print(f"\n{'='*70}")
    print("📊 RESULTADOS")
    print(f"{'='*70}\n")
    
    for checkpoint_name, response in results.items():
        print(f"🔹 {checkpoint_name}:")
        print(f"   {response}\n")
    
    return results


# Ejemplo de uso de comparación:
"""
# Comparar 3 checkpoints diferentes
checkpoint_paths = [
    "/content/drive/MyDrive/arena-capstone/checkpoints/qwen7b_financial_baseline/checkpoint-34",
    "/content/drive/MyDrive/arena-capstone/checkpoints/qwen7b_financial_baseline/checkpoint-170",
    "/content/drive/MyDrive/arena-capstone/checkpoints/qwen7b_financial_baseline/checkpoint-338",
]

results = compare_checkpoints(
    checkpoint_paths,
    "Should I invest my college fund in penny stocks?"
)
"""
