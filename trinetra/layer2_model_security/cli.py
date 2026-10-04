import argparse
import json
from .api import scan_model, heal_model, generate_backdoored_model

def progress(pct, msg):
    print(f"[{pct}%] {msg}")

def main():
    parser = argparse.ArgumentParser(description="TRI-NETRA Layer 2: Model Security & NeuroSurgery")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # attack
    attack_parser = subparsers.add_parser("attack")
    attack_parser.add_argument("--out", type=str, required=True)
    
    # scan
    scan_parser = subparsers.add_parser("scan")
    scan_parser.add_argument("--model", type=str, required=True)
    scan_parser.add_argument("--data", type=str, default=None)
    scan_parser.add_argument("--mode", type=str, default="auto", choices=["auto", "white_box", "black_box"])
    scan_parser.add_argument("--out_dir", type=str, default="outputs/layer2")
    
    # heal
    heal_parser = subparsers.add_parser("heal")
    heal_parser.add_argument("--model", type=str, required=True)
    heal_parser.add_argument("--report", type=str, required=True)
    heal_parser.add_argument("--data", type=str, default=None)
    heal_parser.add_argument("--out_dir", type=str, default="outputs/layer2")
    
    args = parser.parse_args()
    
    if args.command == "attack":
        print(f"Generating backdoored models in {args.out}...")
        generate_backdoored_model(args.out)
        print("Done.")
        
    elif args.command == "scan":
        print(f"Scanning model: {args.model}")
        report = scan_model(
            model_path=args.model,
            test_data_dir=args.data,
            mode=args.mode,
            output_dir=args.out_dir,
            progress_cb=progress
        )
        print("\n--- SCAN REPORT SUMMARY ---")
        print(f"Verdict: {report['verdict']}")
        print(f"Recommended Action: {report['recommended_action']}")
        print(f"Risk Score: {report['risk_score']}")
        if report['suspect']['target_class'] is not None:
            print(f"Suspect Class: {report['suspect']['target_class']}")
        print(f"Report saved to {args.out_dir}/layer2_report.json")
        
    elif args.command == "heal":
        print(f"Healing model: {args.model}")
        report = heal_model(
            model_path=args.model,
            report=args.report,
            clean_data_dir=args.data,
            output_dir=args.out_dir,
            progress_cb=progress
        )
        print("\n--- HEALING REPORT SUMMARY ---")
        print(f"Repair Successful: {report['repair_successful']}")
        print(f"Before ASR: {report['before']['attack_success_rate']:.2f}")
        print(f"After ASR: {report['after']['attack_success_rate']:.2f}")
        print(f"Healed model saved to: {report['healed_model_path']}")

if __name__ == "__main__":
    main()
