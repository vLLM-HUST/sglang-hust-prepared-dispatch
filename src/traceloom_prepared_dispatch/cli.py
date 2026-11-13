import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Build or explicitly mount SGLang prepared dispatch")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="compile and validate in a fresh CPU process on the pinned stack")
    mount = sub.add_parser("mount", help="mount into a dedicated pinned SGLang checkout")
    mount.add_argument("source")
    mount.add_argument("--with-batch-copy", action="store_true")
    unmount = sub.add_parser("unmount", help="restore original source bytes after stopping the engine")
    unmount.add_argument("source")
    sub.add_parser("check", help="verify runtime, schema and native build identity without model loading")
    args = parser.parse_args()
    if args.command == "mount":
        from .mount import mount
        result = mount(args.source, with_batch_copy=args.with_batch_copy)
    elif args.command == "unmount":
        from .mount import unmount
        result = unmount(args.source)
    elif args.command == "build":
        from .native import build
        result = build()
    else:
        from .native import check_runtime, load_native
        result = check_runtime()
        load_native()
        result["native_schema_validated"] = True
    print(json.dumps(result, indent=2))
