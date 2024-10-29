I followed Black Forest Labs' official implementation on [here](https://github.com/black-forest-labs/flux/tree/main/src/flux)
# Flux Image Generations

Default Generation Settings:
```python
width: int = 512
height: int = 512
num_steps: int = 4
guidance: float = 3.5
seed: int = 1
```

## Model Details
- Default model: Flux-Schnell
- Alternative option: Change to Flux-dev by modifying:
```python
flux_model_name = "flux-dev"
```

## Generated Images

### 1. Desert Cat
![Cat in desert](flux_generated/catdesert.webp)

**Prompt:**
"Cat in the middle of desert."

---

### 2. Pumpkin Cat
![VenomSpiderman](flux_generated/venom-spiderman.webp)

**Prompt:**
"Venom cuddles spiderman"
