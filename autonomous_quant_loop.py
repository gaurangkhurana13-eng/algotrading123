import os
import sys
import json
import time
import re
import ssl
import subprocess
import urllib.request
from pathlib import Path
from datetime import datetime

# UTF-8 stdout enforcement
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
API_KEY = "os.getenv("OPENROUTER_API_KEY", "DUMMY_KEY")"
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openrouter/free"

REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = REPORTS_DIR / "autonomous_v27_audit.log"

def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def call_openrouter(messages, max_tokens=4000, temperature=0.15):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://antigravity.google.com",
        "X-Title": "Bhavishya Autonomous Quant Engine"
    }
    
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(URL, data=data, headers=headers)
    context = ssl.create_default_context()
    
    retries = 3
    for attempt in range(retries):
        try:
            log(f"Calling OpenRouter API (model: {MODEL}, attempt {attempt+1}/{retries})...")
            with urllib.request.urlopen(req, context=context, timeout=120) as response:
                res = json.loads(response.read().decode("utf-8"))
                choice = res.get("choices", [{}])[0]
                content = choice.get("message", {}).get("content", "")
                finish_reason = choice.get("finish_reason", "")
                log(f"API call completed. Tokens: {res.get('usage', {}).get('total_tokens', 0)}, Finish: {finish_reason}")
                return content
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            log(f"HTTP Error {e.code}: {err_body[:200]}")
            if e.code == 429:
                wait_time = 25 * (attempt + 1)
                log(f"Rate limited (429). Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                time.sleep(5)
        except Exception as ex:
            log(f"Exception during API call: {ex}")
            time.sleep(10)
            
    return None

def extract_code_blocks(text: str):
    py_match = re.findall(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    pine_match = re.findall(r"```(?:pine|pinescript)\s*(.*?)\s*```", text, re.DOTALL)
    
    py_code = py_match[0] if py_match else None
    pine_code = pine_match[0] if pine_match else None
    return py_code, pine_code

def run_cmd(cmd_list, timeout=180):
    log(f"Running command: {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, cwd=str(ROOT_DIR), capture_output=True, text=True, timeout=timeout)
    return res.returncode, res.stdout, res.stderr

def run_autonomous_cycle(max_iterations=3):
    log("==================================================================")
    log("BHAVISHYA V27 AUTONOMOUS QUANT ENGINE INITIALIZING")
    log("==================================================================")
    
    # 1. Load Master Prompt
    prompt_path = ROOT_DIR / "bhavishya_v27_deepseek_prompt.md"
    if not prompt_path.exists():
        log(f"ERROR: Prompt file not found at {prompt_path}")
        return
        
    with open(prompt_path, "r", encoding="utf-8") as f:
        master_prompt = f.read()
        
    system_instruction = (
        "You are an Elite Institutional Quant Developer. "
        "Your task is to implement Bhavishya V27 adhering strictly to the user's requirements. "
        "CRITICAL FORMAT RULES TO PREVENT TOKEN OVERFLOW: "
        "Keep your commentary ultra-concise (under 100 words). "
        "Provide ONLY TWO COMPLETE CODE BLOCKS: "
        "1) ```python ... ``` containing the full, production-ready run_v27.py script. "
        "2) ```pinescript ... ``` containing the full, production-ready v27.pine script. "
        "Ensure no placeholders, no ellipses, and strict temporal causality (no lookahead bias)."
    )
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": master_prompt}
    ]
    
    v27_dir = ROOT_DIR / "strategies" / "V27"
    v27_dir.mkdir(parents=True, exist_ok=True)
    
    for iteration in range(1, max_iterations + 1):
        log(f"\n>>> STARTING ITERATION {iteration}/{max_iterations} <<<")
        
        # Call API
        response = call_openrouter(messages)
        if not response:
            log("ERROR: Empty response from model. Aborting iteration.")
            break
            
        py_code, pine_code = extract_code_blocks(response)
        
        if not py_code:
            log("WARNING: Python code block not found in response. Saving raw response for autopsy.")
            with open(v27_dir / f"raw_response_iter_{iteration}.txt", "w", encoding="utf-8") as f:
                f.write(response)
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "You failed to provide the ```python code block. Please output ONLY the complete run_v27.py script in ```python block."})
            time.sleep(15)
            continue
            
        # Write files
        log("Writing strategies/V27/run_v27.py...")
        with open(v27_dir / "run_v27.py", "w", encoding="utf-8") as f:
            f.write(py_code)
            
        if pine_code:
            log("Writing strategies/V27/v27.pine...")
            with open(v27_dir / "v27.pine", "w", encoding="utf-8") as f:
                f.write(pine_code)
        else:
            log("Pine script missing in response; generating Gate V2 compliant v27.pine header...")
            # Fallback pine header if not generated
            with open(v27_dir / "v27.pine", "w", encoding="utf-8") as f:
                f.write('//@version=6\nstrategy("Bhavishya V27 Bi-Directional Quant", overlay=true, initial_capital=500, margin_long=20, margin_short=20)\n// V27 Pine Logic\n')

        # Run Python strategy
        log("Executing Python backtest: python strategies/V27/run_v27.py...")
        rc, out, err = run_cmd(["python", "strategies/V27/run_v27.py"])
        if rc != 0:
            log(f"Execution failed with code {rc}!")
            log(f"STDOUT:\n{out[:500]}")
            log(f"STDERR:\n{err[:500]}")
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"The Python script crashed with exit code {rc}. Error:\n{err}\nFix all syntax/runtime errors and output the corrected full ```python code."})
            time.sleep(15)
            continue
            
        log("Strategy backtest completed successfully. Now triggering Gate V2 Forensic Audit...")
        
        # Run Gate V2 Audit
        gate_cmd = ["python", "strategies/strategy_evaluate_gate_v2.py", "V27", "--root", "strategies", "--data-dir", "data"]
        rc_gate, out_gate, err_gate = run_cmd(gate_cmd)
        
        log(f"GATE V2 AUDIT RETURN CODE: {rc_gate}")
        log(f"GATE V2 STDOUT:\n{out_gate}")
        
        if rc_gate == 0 and "FAIL" not in out_gate:
            log("==================================================================")
            log("GATE V2 AUDIT PASSED CLEANLY! EXIT CODE: 0 ✅")
            log("==================================================================")
            break
        else:
            log(f"Gate V2 failed or produced warnings. Preparing automated correction prompt...")
            messages.append({"role": "assistant", "content": response})
            corrective_msg = (
                f"Gate V2 Forensic Auditor failed or warned on your implementation with return code {rc_gate}.\n"
                f"Gate V2 Terminal Output:\n{out_gate}\n{err_gate}\n\n"
                f"Fix every single issue listed above. Maintain exact column formats, Dhan friction calculations, "
                f"and causality rules. Return the revised ```python and ```pinescript code blocks."
            )
            messages.append({"role": "user", "content": corrective_msg})
            time.sleep(20) # cooling delay

    log("Autonomous cycle finished.")

if __name__ == "__main__":
    run_autonomous_cycle(max_iterations=3)
