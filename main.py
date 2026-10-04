import logging
import argparse
from src.utils.config import load_config, setup_logging
from src.data.provider import YFinanceHistoricalProvider
from src.data.fetcher import DataIngester

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="DL Stock Risk Prediction CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")
    
    # Ingest Command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest historical market data")
    ingest_parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g. RELIANCE.NS, ^NSEI)")
    ingest_parser.add_argument("--start", help="Start date (YYYY-MM-DD)", default=None)
    ingest_parser.add_argument("--end", help="End date (YYYY-MM-DD)", default=None)
    
    # Features Command
    feat_parser = subparsers.add_parser("features", help="Generate technical features")
    feat_parser.add_argument("--ticker", required=True, help="Ticker symbol to process")
    
    # Targets Command
    targ_parser = subparsers.add_parser("targets", help="Generate strictly isolated targets")
    targ_parser.add_argument("--ticker", required=True, help="Ticker symbol to process")
    
    # Splits/Validation Command
    # Train Command
    # Ablation Command
    # Explain Command
    # Backtest Command
    # Infer Command
    infer_parser = subparsers.add_parser("infer", help="Run live market data inference loop")
    infer_parser.add_argument("--ticker", required=True, help="Ticker symbol to process")
    infer_parser.add_argument("--model", required=True, choices=["mlp"], help="Model to use for inference")
    infer_parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds")
    infer_parser.add_argument("--iterations", type=int, default=3, help="Number of times to poll before exiting (for testing)")
    backtest_parser = subparsers.add_parser("backtest", help="Run historical backtest on model predictions")
    backtest_parser.add_argument("--ticker", required=True, help="Ticker symbol to process")
    backtest_parser.add_argument("--model", required=True, choices=["majority", "logistic", "random_forest", "xgboost", "mlp", "lstm"], help="Model predictions to backtest")
    backtest_parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for long entry")
    explain_parser = subparsers.add_parser("explain", help="Run model explainability (Permutation/SHAP)")
    explain_parser.add_argument("--ticker", required=True, help="Ticker symbol to process")
    explain_parser.add_argument("--model", required=True, choices=["mlp"], help="Model to explain")
    ablation_parser = subparsers.add_parser("ablation", help="Run feature ablation experiments")
    ablation_parser.add_argument("--ticker", required=True, help="Ticker symbol to process")
    train_parser = subparsers.add_parser("train", help="Train baseline or DL models")
    train_parser.add_argument("--ticker", required=True, help="Ticker symbol to process")
    train_parser.add_argument("--model", required=True, choices=["majority", "logistic", "random_forest", "xgboost", "mlp", "lstm"], help="Model to train")
    split_parser = subparsers.add_parser("splits", help="Validate and visualize chronological splits")
    split_parser.add_argument("--ticker", required=True, help="Ticker symbol to process")
    
    args = parser.parse_args()
    config = load_config()
    
    if args.command == "ingest":
        start = args.start or config['data']['start_date']
        end = args.end or config['data']['end_date']
        
        logger.info("Initializing ingestion pipeline...")
        provider = YFinanceHistoricalProvider(retries=3)
        ingester = DataIngester(provider=provider, config=config)
        
        ingester.ingest(args.ticker, start, end)
        
    elif args.command == "features":
        import pandas as pd
        from src.utils.config import PROJECT_ROOT
        from src.features.generator import FeatureGenerator
        
        logger.info(f"Initializing feature generation for {args.ticker}...")
        raw_dir = PROJECT_ROOT / config['data'].get('raw_dir', 'data/raw')
        
        files = list(raw_dir.glob(f"{args.ticker}_*.csv"))
        if not files:
            logger.error(f"No raw data found for {args.ticker}. Run ingest first.")
            return
        latest_file = sorted(files)[-1]
        
        df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
        generator = FeatureGenerator(config)
        df_features = generator.generate(df, context_dfs={})
        
        proc_dir = PROJECT_ROOT / config['data'].get('processed_dir', 'data/processed')
        proc_dir.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(proc_dir / f"{args.ticker}_features.csv")
        logger.info(f"Features saved to {proc_dir / f'{args.ticker}_features.csv'}")

    elif args.command == "targets":
        import pandas as pd
        from src.utils.config import PROJECT_ROOT
        from src.targets.generator import TargetGenerator
        
        logger.info(f"Initializing target generation for {args.ticker}...")
        proc_dir = PROJECT_ROOT / config['data'].get('processed_dir', 'data/processed')
        feature_path = proc_dir / f"{args.ticker}_features.csv"
        
        if not feature_path.exists():
            logger.error(f"Features file not found at {feature_path}. Run features first.")
            return
            
        df_features = pd.read_csv(feature_path, index_col=0, parse_dates=True)
        generator = TargetGenerator(config)
        
        X, y = generator.generate(df_features, drop_na_targets=True)
        
        X.to_csv(proc_dir / f"{args.ticker}_X.csv")
        y.to_csv(proc_dir / f"{args.ticker}_y.csv")
        logger.info(f"Isolated Feature Matrix (X) saved to {proc_dir / f'{args.ticker}_X.csv'}")
        logger.info(f"Target Vector (y) saved to {proc_dir / f'{args.ticker}_y.csv'}")
        
    elif args.command == "splits":
        import pandas as pd
        from src.utils.config import PROJECT_ROOT
        from src.evaluation.splitter import TimeSeriesSplitter
        
        logger.info(f"Initializing validation split framework for {args.ticker}...")
        proc_dir = PROJECT_ROOT / config['data'].get('processed_dir', 'data/processed')
        reports_dir = PROJECT_ROOT / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            X = pd.read_csv(proc_dir / f"{args.ticker}_X.csv", index_col=0, parse_dates=True)
            y = pd.read_csv(proc_dir / f"{args.ticker}_y.csv", index_col=0, parse_dates=True)
        except FileNotFoundError:
            logger.error("X.csv or y.csv not found. Run 'targets' first.")
            return
            
        splitter = TimeSeriesSplitter(config)
        
        # 1. 70/15/15 Plot
        logger.info("Verifying standard 70/15/15 chronological isolation...")
        plot_path = reports_dir / f"{args.ticker}_split_visualization.png"
        splitter.plot_splits(X, save_path=plot_path)
        
        # 2. Walk Forward
        wf_splits = splitter.walk_forward_splits(X, y, n_splits=5)
        splitter.log_walk_forward_table(wf_splits)

    elif args.command == "explain":
        from src.utils.config import PROJECT_ROOT
        from src.explainability.explainer import ModelExplainer
        import torch
        import pandas as pd
        from src.models.dl_models import ConfigurableMLP
        from src.evaluation.splitter import TimeSeriesSplitter
        
        logger.info(f"Initializing explainability for {args.ticker} {args.model.upper()}...")
        proc_dir = PROJECT_ROOT / config['data'].get('processed_dir', 'data/processed')
        models_dir = PROJECT_ROOT / 'models'
        reports_dir = PROJECT_ROOT / 'reports'
        
        X = pd.read_csv(proc_dir / f"{args.ticker}_X.csv", index_col=0, parse_dates=True)
        y = pd.read_csv(proc_dir / f"{args.ticker}_y.csv", index_col=0, parse_dates=True)["Target_Direction"]
        
        splitter = TimeSeriesSplitter(config)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = splitter.split_train_val_test(X, y)
        scaler, X_train_s, X_val_s, X_test_s = splitter.scale_features(X_train, X_val, X_test)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        dl_cfg = config["models"].get("deep_learning", {})
        layers = dl_cfg.get("layers", [128, 64, 32])
        dropout = dl_cfg.get("dropout", 0.3)
        
        model = ConfigurableMLP(input_dim=X.shape[1], layer_dims=layers, dropout=dropout).to(device)
        model_path = models_dir / f"{args.ticker}_{args.model}_Target_Direction.pt"
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            explainer = ModelExplainer(model, device, X.columns.tolist(), reports_dir, args.ticker, args.model)
            explainer.run_permutation_importance(X_test_s.values, y_test.values)
            explainer.run_shap(X_train_s.values, X_test_s.values)
            
            logger.info("Explainability complete. Review outputs in reports/ directory.")
        except Exception as e:
            import traceback
            logger.error(f"Explainability failed: {e}\n{traceback.format_exc()}")

    elif args.command == "infer":
        from src.utils.config import PROJECT_ROOT
        from src.inference.engine import InferenceEngine
        from src.data.live_provider import YFinanceLiveProvider
        
        provider = YFinanceLiveProvider()
        engine = InferenceEngine(config, provider, PROJECT_ROOT)
        
        try:
            engine.run_live_inference(args.ticker, args.model)
        except Exception as e:
            import traceback
            logger.error(f"Inference failed: {e}\n{traceback.format_exc()}")

    elif args.command == "backtest":
        from src.utils.config import PROJECT_ROOT
        from src.backtesting.engine import Backtester
        logger.info(f"Initializing Backtester for {args.ticker} {args.model.upper()}...")
        try:
            bt = Backtester(config, PROJECT_ROOT)
            bt.run(args.ticker, args.model, target_col="Target_Direction", threshold=args.threshold)
            logger.info("Backtest execution completed. View reports/ for equity curve.")
        except Exception as e:
            import traceback
            logger.error(f"Backtest failed: {e}\n{traceback.format_exc()}")

    elif args.command == "ablation":
        from src.experiments.feature_ablation import run_ablation
        logger.info(f"Starting feature ablation experiment for {args.ticker}...")
        try:
            run_ablation(args.ticker)
            logger.info("Ablation experiment completed successfully.")
        except Exception as e:
            import traceback
            logger.error(f"Ablation experiment failed: {e}\n{traceback.format_exc()}")

    elif args.command == "train":
        from src.utils.config import PROJECT_ROOT
        
        if args.model == "mlp":
            from src.models.dl_trainer import DLTrainer
            trainer = DLTrainer(config, PROJECT_ROOT)
            try:
                metrics = trainer.train_mlp(args.ticker, target_col='Target_Direction')
            except Exception as e:
                import traceback
                logger.error(f"DL Training failed: {e}\n{traceback.format_exc()}")
        elif args.model == "lstm":
            from src.models.dl_trainer import DLTrainer
            trainer = DLTrainer(config, PROJECT_ROOT)
            try:
                metrics = trainer.train_lstm(args.ticker, target_col='Target_Direction')
            except Exception as e:
                import traceback
                logger.error(f"LSTM Training failed: {e}\n{traceback.format_exc()}")
        else:
            from src.models.trainer import ModelTrainer
            trainer = ModelTrainer(config, PROJECT_ROOT)
            try:
                metrics = trainer.train_baseline(args.ticker, args.model, target_col='Target_Direction')
                val_metrics = metrics['metrics']['val']
                logger.info(f"=== {args.model.upper()} Validation Performance ===")
                logger.info(f"{'Metric':<20} | {'Score'}")
                logger.info("-" * 30)
                for k, v in val_metrics.items():
                    if k != "Confusion Matrix" and v is not None:
                        logger.info(f"{k:<20} | {v:.4f}")
                logger.info(f"Confusion Matrix: {val_metrics['Confusion Matrix']}")
            except Exception as e:
                import traceback
                logger.error(f"Training failed: {e}\n{traceback.format_exc()}")

    elif args.command is None:
        logger.info("Deep Learning Based Stock Risk Prediction System initialized.")
        logger.info("Use 'python main.py -h' to see available commands.")

if __name__ == "__main__":
    main()
