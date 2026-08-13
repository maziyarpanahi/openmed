{
  description = "OpenMed reproducible package and development shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs =
    { self, nixpkgs }:
    let
      supportedSystems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python312;
          version = builtins.elemAt (
            builtins.match
              ''.*__version__ = "([^"]+)".*''
              (builtins.replaceStrings [ "\n" ] [ " " ] (builtins.readFile ./openmed/__about__.py))
          ) 0;
        in
        rec {
          openmed = python.pkgs.buildPythonPackage {
            pname = "openmed";
            inherit version;
            pyproject = true;

            src = self;

            build-system = [ python.pkgs.hatchling ];
            dependencies = with python.pkgs; [
              faker
              jieba
              pysbd
              pyyaml
            ];

            # The full suite runs in the development shell in nix.yml. Keeping
            # package checks import-only avoids pulling development tools into
            # the runtime closure.
            doCheck = false;
            pythonImportsCheck = [ "openmed" ];

            meta = {
              description = "Local-first clinical NLP and de-identification toolkit";
              homepage = "https://github.com/maziyarpanahi/openmed";
              license = pkgs.lib.licenses.asl20;
              mainProgram = "openmed";
            };
          };

          default = openmed;
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python312;
          openmed = self.packages.${system}.openmed;

          # The generated service stubs require grpcio >= 1.81.1. Keep the
          # runtime and code generator paired until that patch release lands
          # in the pinned nixpkgs branch.
          grpcio = python.pkgs.grpcio.overridePythonAttrs (_: {
            version = "1.81.1";
            src = pkgs.fetchPypi {
              pname = "grpcio";
              version = "1.81.1";
              hash = "sha256-b6EKdnFDpegujqq1ORivDNiQmleif4yyKIuAphOsZxs=";
            };
          });
          grpcio-tools = python.pkgs.grpcio-tools.overridePythonAttrs (_: {
            version = "1.81.1";
            src = pkgs.fetchPypi {
              pname = "grpcio_tools";
              version = "1.81.1";
              hash = "sha256-oio4cBgJJ/3YTisn0HnvW39fjGEQGBtnNq/BekY0gfE=";
            };
            dependencies = [
              grpcio
              python.pkgs.protobuf
              python.pkgs.setuptools
            ];
          });
          # Keep Nix's development shell inside the version bounds declared
          # by the canonical dev extra. The pinned nixpkgs revision trails
          # both packages, while the tests exercise their current APIs.
          mcp = python.pkgs.mcp.overridePythonAttrs (_: {
            version = "1.27.1";
            src = pkgs.fetchFromGitHub {
              owner = "modelcontextprotocol";
              repo = "python-sdk";
              tag = "v1.27.1";
              hash = "sha256-LhoLcFC5+7xOCfud23sbHyTMxKYmdeZh0c+UtGdvzCs=";
            };
          });
          crossWeb = python.pkgs.cross-web.overridePythonAttrs (_: {
            version = "0.7.0";
            src = pkgs.fetchPypi {
              pname = "cross_web";
              version = "0.7.0";
              hash = "sha256-FfvIuagkoFXbgSf9bkPgdzB09iD97LaytYfT0KK91Fk=";
            };
            dependencies = [ python.pkgs.typing-extensions ];
            doCheck = false;
            nativeCheckInputs = [ ];
          });
          strawberryGraphql =
            python.pkgs.strawberry-graphql.overridePythonAttrs (_: {
              version = "0.319.0";
              src = pkgs.fetchFromGitHub {
                owner = "strawberry-graphql";
                repo = "strawberry";
                tag = "0.319.0";
                hash = "sha256-7mbinSIb0AhqMggaziiLCZQBJ0i2G6Dq0ZjGVnFLDiY=";
              };
              postPatch = ''
                substituteInPlace pyproject.toml \
                  --replace-fail 'version = "0.318.1"' 'version = "0.319.0"'
                substituteInPlace pyproject.toml \
                  --replace-fail "uv_build>=0.11,<0.12" "uv_build"
                substituteInPlace pyproject.toml \
                  --replace-fail "--emoji" ""
              '';
              build-system = [ python.pkgs.uv-build ];
              dependencies = with python.pkgs; [
                crossWeb
                graphql-core
                packaging
                python-dateutil
                typing-extensions
              ];

              # The OpenMed suite runs after entering the dev shell. Avoid
              # Strawberry's upstream optional-integration matrix here: it
              # pulls every supported web framework into the build graph and
              # makes shell construction depend on their unrelated test
              # suites (including Sanic's timing-sensitive network tests).
              doCheck = false;
              nativeCheckInputs = [ ];
            });

          devPythonPackages =
            (with python.pkgs; [
              brotli
              cryptography
              dask
              duckdb
              fastapi
              fonttools
              fsspec
              huggingface-hub
              httpx
              hypothesis
              jsonschema
              mypy
              numpy
              opencc
              openpyxl
              opentelemetry-api
              opentelemetry-exporter-otlp-proto-http
              opentelemetry-sdk
              pandas
              pillow
              pyarrow
              polars
              protobuf
              pytest
              pytest-cov
              pytest-timeout
              python-dateutil
              python-multipart
              rich
              sqlalchemy
              typer
            ])
            ++ [
              grpcio
              grpcio-tools
              mcp
              openmed
              strawberryGraphql
            ];
          pythonPath = python.pkgs.makePythonPath devPythonPackages;
        in
        {
          default = pkgs.mkShell {
            packages =
              [
                python
                pkgs.pre-commit
                pkgs.ruff
              ]
              ++ devPythonPackages;

            # Use the unwrapped interpreter so subprocess sandbox checks see
            # the real standard-library prefix rather than a symlink farm.
            shellHook = ''
              export PYTHONPATH="${pythonPath}''${PYTHONPATH:+:}$PYTHONPATH"
            '';
          };
        }
      );

      checks = forAllSystems (system: {
        package = self.packages.${system}.openmed;
        dev-shell = self.devShells.${system}.default;
      });
    };
}
