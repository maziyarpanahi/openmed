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

          devPythonPackages =
            (with python.pkgs; [
              dask
              duckdb
              fastapi
              fsspec
              httpx
              hypothesis
              jsonschema
              mypy
              numpy
              opentelemetry-api
              opentelemetry-exporter-otlp-proto-http
              opentelemetry-sdk
              pandas
              pyarrow
              polars
              protobuf
              pytest
              pytest-cov
              python-dateutil
            ])
            ++ [
              grpcio
              grpcio-tools
              openmed
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
