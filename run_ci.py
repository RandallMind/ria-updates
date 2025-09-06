#!/usr/bin/env python3
import subprocess, json, traceback, sys, os

logfile = "updates_wrapper.log"
def write_json_error(exc_text):
    try:
        err = {"error": True, "message": str(exc_text), "traceback": traceback.format_exc()}
        with open("updates.json", "w", encoding="utf-8") as f:
            json.dump(err, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

try:
    # Ejecutar el script original y capturar salida
    p = subprocess.run([sys.executable, "update_sweep.py"], capture_output=True, text=True)
    with open(logfile, "w", encoding="utf-8") as f:
        f.write("STDOUT:\\n")
        f.write(p.stdout or "")
        f.write("\\n\\nSTDERR:\\n")
        f.write(p.stderr or "")
    if p.returncode != 0:
        write_json_error(p.stderr or p.stdout or "Non-zero exit")
        sys.exit(p.returncode)
    # Si el script terminó bien pero no produjo updates.json, escribimos aviso
    if not os.path.exists("updates.json"):
        with open("updates.json", "w", encoding="utf-8") as f:
            json.dump({"warning": "script finished but updates.json missing", "stdout": p.stdout[:1000]}, f, ensure_ascii=False, indent=2)
except Exception as e:
    write_json_error(str(e))
    raise
