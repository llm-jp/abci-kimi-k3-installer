# ABCIでKimi K3を動かす

ABCIのH200搭載ノード向けに、Singularityを使わないSGLang実行環境を構築します。
1レプリカは2ノード、H200×16、TP16、EP16で構成します。
複数レプリカはSGLang Routerで束ね、単一のOpenAI互換エンドポイントとして
公開します。

インストールジョブはGitHub、PyPI、PyTorch、Rustup、crates.ioから依存関係を
取得します。Kimi K3のモデルの重みは取得しません。

## 前提

全計算ノードから参照できる共有ストレージに、Kimi K3のモデルの重みを
配置してください。重みは本インストーラに含まれません。

次の構成はスクリプト内で固定しています。

- インストール: Spotサービスの`rt_HC`を1ノード
- 推論サーバー: Spotサービスの`rt_HF`を1レプリカあたり2ノード
  （各ノードH200×8）
- GCC 13.2.0、CUDA 13.0.1、Python 3.12.9、NCCL 2.28.3、HPC-X 2.26
- SGLangのコミット`9cb03516b2baa9b42a418de98deea491a9ab8eb9`
- Rust 1.90.0、SGLang Router 0.3.2
- TP16、EP16、Marlin、FlashMLA、`ibn1`
- SGLang Routerの振り分け方式: `cache_aware`

## 1. 設定

このディレクトリをABCI上に配置し、`config.env`を作成します。

```bash
cd ~/abci-kimi-k3-installer
cp config.example.env config.env
```

編集する値は次の3項目だけです。

```bash
ABCI_PROJECT="ABCIグループ名"
MODEL_DIR="/共有パス/MoonshotAI/Kimi-K3"
RUNTIME_ROOT="/共有パス/Kimi-K3-runtime"
```

`MODEL_DIR`には`config.json`、`model.safetensors.index.json`、インデックスが
参照するすべての重みシャードが必要です。`RUNTIME_ROOT`には実行環境を
インストールする、まだ存在しない共有ストレージ上のパスを指定します。

## 2. インストール

ABCIのログインノードで実行します。

```bash
bash submit-install.sh
```

表示されたジョブIDを`qstat`で確認します。成功時には`K3_INSTALL_OK`を
表示します。`RUNTIME_ROOT`が既に存在する場合は、ジョブを投入せず停止します。

構築したPythonパッケージの一覧は次のファイルに保存します。

```text
RUNTIME_ROOT/environment.freeze.txt
```

## 3. 起動

まず1レプリカで起動します。

```bash
bash submit-server.sh 1
```

起動時にモデルの重みを読み込み、CUDAグラフを作成するため、完了まで十数分
かかる場合があります。起動が完了すると次を表示します。

```text
K3_ROUTED_REPLICA_POOL_READY count=1
router_ssh_host=HOST
router_api_base=http://NODE_IP:31000/v1
```

`router_ssh_host`の値を使ってAPIを確認します。

```bash
bash client.sh smoke HOST
```

成功時には`K3_ROUTER_API_OK`を表示します。

Nレプリカで起動する場合は次のとおりです。

```bash
bash submit-server.sh N
```

必要なリソースは`2 × N`ノード、`16 × N`GPUです。スクリプト独自の
レプリカ数上限はありません。実際に投入できる数は、ABCIのSpotサービスの
リソース制限で決まります。

## 4. トークン上限と性能の確認

起動したレプリカ数を指定し、入力トークン数の実効上限を確認します。

```bash
bash client.sh capacity HOST REPLICA_COUNT
```

サーバー数ごとのスループットを測定します。

```bash
bash client.sh benchmark \
  HOST \
  REPLICA_COUNT \
  CONCURRENCY_PER_SERVER \
  WAVES \
  INPUT_TOKENS \
  OUTPUT_TOKENS \
  SEED
```

各サーバーへ同数のリクエストを直接送り、1台構成から全台構成まで測定します。
結果はJSON形式で`results/`へ保存します。

## 停止とログ

サーバーは、PBSの実行時間上限への到達、プロセスの異常終了、または`qdel`に
よる停止のいずれかまで稼働します。

```bash
qdel JOB_ID
```

サーバーログは、ジョブ投入元ディレクトリの
`k3-Nreplicas-JOB_NUMBER/`へ保存します。主な確認先は`router.log`と
`replica-N/node-rank-*.log`です。
