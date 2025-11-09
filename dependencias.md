# Dependências do Ambiente Valentina Moreau

Este documento consolida todos os componentes que precisam ser instalados ou configurados para preparar o ambiente de desenvolvimento, treinamento e inferência descrito no plano do projeto.

## 1. Requisitos de Sistema
- **Sistema operacional:** Linux 64 bits com suporte a drivers NVIDIA recentes (Ubuntu 22.04 LTS recomendado para compatibilidade com CUDA).
- **GPU:** A100 80GB (preferencial) ou GPU equivalente com suporte total a CUDA 11.8+ para treinamento e inferência intensivos.
- **CPU e memória:** CPU x86_64 multicore com pelo menos 32 GB de RAM para pré-processamento e suporte às cargas de trabalho de inferência.
- **Armazenamento:** Acesso a storage persistente (S3 ou GCS) para datasets, checkpoints e artefatos de experimentos.

## 2. Contas e Serviços na Nuvem
- **Modal.com:** Conta ativa com tokens/secrets configurados para executar jobs de treinamento, inferência e workflows automatizados.
- **Provedor de storage (S3 ou GCS):** Bucket dedicado para armazenar dados de treino, checkpoints, metadados e resultados.
- **Plataforma de rastreamento de experimentos:** Weights & Biases ou MLflow para registro de hiperparâmetros e métricas.

## 3. Ferramentas de Linha de Comando
- `git` para versionamento de código.
- `git-lfs` para versionar checkpoints e datasets grandes.
- `modal` CLI para gerenciamento de imagens, funções, schedules e workflows na Modal.com.
- `awscli` ou `gcloud` (dependendo do provedor escolhido) para interação com o storage externo.

## 4. Ambiente Python
- **Versão recomendada:** Python 3.10 ou superior (compatível com PyTorch e Diffusers recentes).
- **Gerenciador de pacotes:** `pip` ou ambiente virtual (venv, poetry ou conda) conforme preferência da equipe.

### 4.1 Bibliotecas Principais de IA
- `torch` e `torchvision` (compilados com suporte CUDA 11.8+).
- `diffusers`
- `accelerate`
- `xformers`
- `transformers`
- `safetensors`
- `peft`
- `controlnet-aux` (para suporte a ControlNet/Adapters quando necessário).

### 4.2 Treinamento Personalizado
- `dreambooth` scripts baseados em Diffusers (pode-se utilizar implementações externas, mas garantir dependências como `gradio` ou `tyro` se aplicável).
- `datasets` para manipulação de metadados e prompts.
- `opencv-python` e `albumentations` para pré-processamento de imagens.

### 4.3 Avaliação e Observabilidade
- `clip` ou `open-clip-torch` para cálculo de similaridade.
- `facenet-pytorch` ou `insightface` para embeddings faciais e verificação de identidade.
- SDK da plataforma de rastreamento escolhida (`wandb` ou `mlflow`).

### 4.4 Utilidades Gerais
- `boto3` ou `google-cloud-storage` para integração com o bucket de dados.
- `pillow`
- `numpy`
- `scipy`
- `tqdm`
- `rich` para logs aprimorados.

## 5. Dependências de Sistema Adicionais
- Drivers NVIDIA e toolkit CUDA 11.8 (ou superior compatível) instalados no host ou imagem base.
- `git`/`curl`/`wget` para automações de build das imagens Modal.
- Ferramentas de compressão (`zip`, `unzip`, `tar`) para manipular datasets.

## 6. Configurações Modal.com
- **modal.Image:** Deve incluir todas as dependências Python listadas acima, além do CUDA runtime e bibliotecas do sistema necessárias para PyTorch.
- **modal.NetworkFileSystem** ou integração direta com S3/GCS para leitura e escrita de dados.
- Definição de secrets (tokens Modal, credenciais AWS/GCP, chaves W&B/MLflow) dentro do dashboard da Modal.

## 7. Boas Práticas Complementares
- Fixar versões críticas (PyTorch, Diffusers, Xformers, Transformers) para garantir reprodutibilidade.
- Utilizar arquivos `.env` ou gerenciadores de secrets para credenciais sensíveis.
- Configurar scripts de automação para geração de lotes de prompts de validação e comparação automática dos resultados.

> Atualize este documento conforme novas dependências ou serviços forem adicionados ao projeto.
