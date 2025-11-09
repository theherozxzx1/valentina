# Valentina

Configuração inicial do ambiente e ferramenta base para o projeto Valentina Moreau.

## Pré-requisitos
- Python 3.10+
- [Poetry](https://python-poetry.org/) 1.5 ou superior
- Drivers NVIDIA/CUDA instalados na máquina host quando for utilizar GPU

## Primeiros passos
1. Clone o repositório e acesse o diretório do projeto:
   ```bash
   git clone <repo> && cd valentina
   ```
2. Crie o arquivo `.env` com as variáveis sensíveis necessárias (veja o modelo em [.env.example](.env.example)).
3. Instale as dependências usando Poetry:
   ```bash
   poetry install --with dev
   ```
4. Ative o shell virtual:
   ```bash
   poetry shell
   ```
5. Valide a instalação exibindo a versão da ferramenta:
   ```bash
   valentina --version
   ```

## Scripts úteis
- `make install`: instala as dependências na máquina local.
- `make lint`: executa o Ruff nas verificações estáticas.
- `make format`: aplica Black na base de código.
- `make test`: roda a suíte de testes com Pytest.

## Estrutura do Projeto
```
├── docs/                     # Documentação e plano de projeto
├── src/valentina/            # Código fonte do pacote principal
├── pyproject.toml            # Configuração de dependências (Poetry)
├── README.md                 # Guia rápido de configuração
└── Makefile                  # Atalhos para rotinas comuns de desenvolvimento
```

## Próximos passos
- Preencher scripts de treinamento/inferência dentro do pacote `valentina`.
- Configurar jobs no Modal.com utilizando as dependências já descritas.
- Documentar processos adicionais na pasta `docs/` conforme o projeto evoluir.
