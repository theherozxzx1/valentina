# Valentina Moreau Project Plan

## 1. Visão Geral do Projeto
- **Objetivo principal:** Desenvolver um sistema capaz de gerar imagens altamente fotorrealistas da personagem virtual Valentina Moreau, garantindo consistência de identidade em múltiplos contextos.
- **Abordagem central:** Treinamento de modelos de difusão text-to-image especializados (ex.: Stable Diffusion XL ou SD 1.5) com fine-tuning personalizado (LoRA/DreamBooth) para capturar a identidade da Valentina.
- **Plataforma de execução:** Modal.com para treinar e servir modelos com GPUs sob demanda.

### 1.1 Descrição Visual Base da Valentina Moreau
- **Referência canônica:** Fotografia fornecida pelo cliente (mulher em vestido amarelo claro sentada em quarto iluminado).
- **Características faciais:** Rosto oval, sorriso pronunciado, nariz delicado, lábios cheios com contorno definido, bochechas levemente marcadas. Olhos castanhos médios com sobrancelhas espessas e bem delineadas, expressando sensualidade confiante.
- **Cabelo:** Castanho-escuro, comprimento médio/longo (altura do busto), textura lisa a levemente ondulada, repartido lateralmente.
- **Pele e tonalidade:** Tom bronzeado médio com acabamento homogêneo e impecável, livre de tatuagens ou marcas; destacar iluminação quente para realçar brilho natural.
- **Constituição corporal:** Corpo atlético e curvilíneo, com seios fartos, cintura fina e quadris bem definidos; postura ereta que ressalta feminilidade e magnetismo.
- **Elementos distintivos:** Pele limpa sem tatuagens, presença ocasional de joias discretas (ex.: braceletes finos) para reforçar sofisticação.
- **Estilo predominante:** Look moderno e elegante com peças ajustadas (ex.: vestidos justos de cores suaves), maquiagem natural destacando olhos e lábios para enfatizar sensualidade refinada.
- **Faixa etária aparente:** 25 anos.

## 2. Requisitos e Restrições
- **Qualidade visual:** Imagens 1024×1024 ou superiores, com iluminação e texturas realistas.
- **Consistência de identidade:** Manter traços faciais, tom de pele, cabelo e estilo visual uniformes.
- **Escalabilidade:** Suporte a geração de múltiplas imagens a partir de prompts simultâneos.
- **Latência aceitável:** < 30s por geração em GPUs modernas (A100/RTX 4090 equivalentes).
- **Reprodutibilidade:** Pipelines versionadas, dependências pinadas, dados com controle de versão.
- **Governança:** Garantir direitos de uso dos dados de treinamento e conformidade com políticas de uso ético.

## 3. Coleta e Preparação de Dados
1. **Definição da identidade da Valentina:**
   - Criar um guia visual com descrições e moodboard.
   - Definir elementos fixos (tom de pele, cor/estilo de cabelo, traços faciais, proporções corporais) e variáveis (vestuário, cenários, poses).
2. **Aquisição de imagens base:**
   - Gerar renderizações iniciais ou fotografias de modelo real com autorização, garantindo variedade de poses/expressões.
   - Curadoria manual para remover inconsistências, ângulos extremos ou ruídos.
3. **Anotação e metadados:**
   - Catalogar cada imagem com descritores textuais ricos (prompt ground-truth) e atributos (iluminação, cenário, humor).
   - Armazenar dados e metadados em storage acessível pelo Modal (S3/GCS).
4. **Pré-processamento:**
   - Normalizar resolução (ex.: 1024²), aplicar face alignment quando aplicável.
   - Utilizar ferramentas de limpeza (background removal opcional), compressão sem perdas.

## 4. Arquitetura do Sistema
- **Camada de Treinamento:**
  - Fine-tuning de modelos base (SDXL ou SD 1.5) via técnicas LoRA ou DreamBooth utilizando GPUs na Modal.
  - Automatização com Modal Workflows para agendar jobs de treinamento, checkpoints e avaliações.
- **Camada de Avaliação:**
  - Scripts para gerar lotes de prompts de validação e comparar resultados (CLIP similarity, face embedding matching, avaliações humanas).
- **Camada de Inferência:**
  - API em Modal Functions para geração sob demanda.
  - Suporte a ajustes de parâmetros (guidance scale, steps, seeds) e filtros de segurança.
- **Gerência de Artefatos:**
  - Versionamento de checkpoints em armazenamento externo (S3) com tagging por versão de personagem.
  - Registro de experimentos (Weights & Biases ou MLflow) para rastrear hiperparâmetros.

## 5. Utilização do Modal.com
1. **Infraestrutura:**
   - Criar conta/projeto Modal com credenciais configuradas (tokens e secrets).
   - Definir `modal.Image` com dependências (Python, PyTorch, diffusers, xformers, accelerate).
   - Configurar `modal.NetworkFileSystem` ou conexão com S3 para dados/modelos.
2. **Treinamento:**
   - Implementar Modal `@app.function` para orquestrar fine-tuning (LoRA/DreamBooth) usando scripts baseados em `diffusers`.
   - Habilitar GPUs adequadas (`A100-80GB` recomendado) e auto-scale limitado durante treinamento.
   - Persistir checkpoints em storage ao final de cada job.
3. **Inferência:**
   - Criar endpoint Modal servindo pipeline de geração (carrega modelo base + LoRA).
   - Otimizar com compilação (`torch.compile`), `xformers`, ou ONNX se necessário.
4. **Automação & Observabilidade:**
   - Utilizar Modal Schedules/Workflows para re-treinos periódicos ou fine-tuning incremental.
   - Logar métricas (tempo de geração, consumo) para dashboards externos.

## 6. Estratégia de Modelagem
- **Modelo base:**
  - Avaliar SDXL (melhor qualidade) vs SD 1.5 (menor custo). Iniciar com SDXL Turbo para prototipação rápida, migrando para SDXL full para qualidade final.
- **Técnica de personalização:**
  - Primeira etapa: DreamBooth tradicional com 20-30 imagens bem anotadas para captar identidade.
  - Refinamento: LoRA ou Textual Inversion para ajustes finos em estilos específicos.
  - Considerar ControlNet/ T2I-Adapter para controle de poses/estruturas (ex.: esqueleto openpose, depth maps).
- **Prompt Engineering:**
  - Definir tokens exclusivos (`valentina_moreau`) e prompts padronizados.
  - Criar biblioteca de prompts curados para diferentes cenários (moda, viagem, lifestyle).
- **Consistência Facial:**
  - Incorporar pipelines de face embedding (FaceNet, ArcFace) para medir distância entre faces geradas e identidade alvo.
  - Loop de feedback humano para garantir qualidade.

## 7. Roadmap Inicial
1. **Semana 1:**
   - Consolidar identidade visual, coletar 20-30 imagens base, criar metadados.
   - Preparar ambiente Modal, criar containers base, configurar storage.
2. **Semana 2:**
   - Implementar pipeline de DreamBooth/LoRA no Modal.
   - Executar primeiro treinamento piloto e avaliar resultados.
3. **Semana 3:**
   - Ajustar hiperparâmetros, incorporar avaliação automática.
   - Construir API de inferência inicial.
4. **Semana 4+:**
   - Iterar em qualidade, adicionar ControlNet, otimizações, UI/integrações.

## 8. Riscos e Mitigações
- **Dados insuficientes/variados:** Mitigar com sessões de fotos adicionais ou geração sintética assistida.
- **Sobreajuste (overfitting):** Usar regularização, prompts negativos, validação cruzada com prompts desconhecidos.
- **Custos de GPU elevados:** Utilizar LoRA para reduzir necessidade de re-treinamento completo, aproveitar escalonamento automático do Modal.
- **Inconsistência em produção:** Monitorar outputs, aplicar filtros de qualidade automáticos, fallback para checkpoints estáveis.

## 9. Próximos Passos Imediatos
1. Formalizar conjunto de requisitos visuais com stakeholders.
2. Iniciar coleta e curadoria de imagens para treinamento.
3. Configurar projeto Modal (repos, secrets, storage) e preparar container base com dependências.
4. Montar scripts iniciais para DreamBooth/LoRA usando `diffusers`.

> **Nota:** Este plano deve ser revisitado e atualizado continuamente conforme o projeto evoluir. Sempre consultar este documento antes de implementar mudanças significativas e registrar ajustes relevantes.
