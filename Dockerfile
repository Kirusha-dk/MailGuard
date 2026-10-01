# syntax=docker/dockerfile:1

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

COPY global.json MailGuard.sln ./
COPY src/MailGuard.Core/MailGuard.Core.csproj src/MailGuard.Core/
COPY src/MailGuard.Cli/MailGuard.Cli.csproj src/MailGuard.Cli/
COPY src/MailGuard.Api/MailGuard.Api.csproj src/MailGuard.Api/
COPY src/MailGuard.SelfTest/MailGuard.SelfTest.csproj src/MailGuard.SelfTest/
RUN dotnet restore MailGuard.sln

COPY src/ ./src/
RUN dotnet publish src/MailGuard.Cli/MailGuard.Cli.csproj -c Release -o /out/cli --no-restore \
 && dotnet publish src/MailGuard.Api/MailGuard.Api.csproj -c Release -o /out/api --no-restore \
 && dotnet publish src/MailGuard.SelfTest/MailGuard.SelfTest.csproj -c Release -o /out/selftest --no-restore

FROM mcr.microsoft.com/dotnet/runtime:8.0 AS cli
WORKDIR /workspace
COPY --from=build /out/cli /app
ENTRYPOINT ["dotnet", "/app/MailGuard.Cli.dll"]

FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS api
WORKDIR /app
COPY --from=build /out/api /app
ENV ASPNETCORE_URLS=http://+:8080 \
    DOTNET_EnableDiagnostics=0
EXPOSE 8080
ENTRYPOINT ["dotnet", "MailGuard.Api.dll"]

FROM mcr.microsoft.com/dotnet/runtime:8.0 AS selftest
WORKDIR /app
COPY --from=build /out/selftest /app
ENTRYPOINT ["dotnet", "MailGuard.SelfTest.dll"]

FROM mcr.microsoft.com/dotnet/runtime:8.0 AS benchmark
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY --from=build /out/cli /app
COPY tools/ /workspace/tools/
ENTRYPOINT ["bash", "/workspace/tools/run_public_benchmark_container.sh"]
