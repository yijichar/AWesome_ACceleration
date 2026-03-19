import json, os, re
from tokenizers import Tokenizer as HFTokenizer

class Tokenizer:
  # Harmony special tokens
  START, END, CHANNEL, MESSAGE, RETURN = "<|start|>", "<|end|>", "<|channel|>", "<|message|>", "<|return|>"
  
  def __init__(self, path: str):
    self._tok = HFTokenizer.from_file(os.path.join(path, "tokenizer.json"))
    config = json.load(open(os.path.join(path, "tokenizer_config.json"))) if os.path.exists(os.path.join(path, "tokenizer_config.json")) else {}
    
    # Load generation config for EOS token IDs (reasoning models have multiple stop tokens)
    gen_config_path = os.path.join(path, "generation_config.json")
    gen_config = json.load(open(gen_config_path)) if os.path.exists(gen_config_path) else {}
    
    self.bos_token_id, self.pad_token_id = 199998, 199999
    # Support multiple EOS tokens: <|return|>, <|endoftext|>, <|call|>
    eos_from_config = gen_config.get("eos_token_id", [200002])
    self.eos_token_ids = set(eos_from_config if isinstance(eos_from_config, list) else [eos_from_config])
    self.eos_token_id = 200002  # Primary EOS for compatibility
    
    for tid, info in config.get("added_tokens_decoder", {}).items():
      c = info.get("content")
      if c == "<|startoftext|>": self.bos_token_id = int(tid)
      elif c == "<|return|>": self.eos_token_id = int(tid)
      elif c == "<|endoftext|>": self.pad_token_id = int(tid)

  def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
    return self._tok.encode(text, add_special_tokens=add_special_tokens).ids
  
  def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
    return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

  def apply_chat_template(self, messages: list[dict], reasoning_effort: str = "medium") -> list[int]:
    """Format messages using Harmony protocol matching the official jinja template"""
    import datetime
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # System message matching jinja template exactly
    system = f"""{self.START}system{self.MESSAGE}You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: {date}

Reasoning: {reasoning_effort}

# Valid channels: analysis, commentary, final. Channel must be included for every message.{self.END}"""
    
    formatted = [system]
    for msg in messages:
      role, content = msg.get("role", "user"), msg.get("content", "")
      if role in ("system", "developer"):
        formatted.append(f"{self.START}developer{self.MESSAGE}# Instructions\n\n{content}{self.END}")
      elif role == "user":
        formatted.append(f"{self.START}user{self.MESSAGE}{content}{self.END}")
      elif role == "assistant":
        formatted.append(f"{self.START}assistant{self.CHANNEL}final{self.MESSAGE}{content}{self.END}")
    formatted.append(f"{self.START}assistant")
    return self.encode("".join(formatted), add_special_tokens=False)

  def parse_harmony_output(self, ids: list[int]):
    """Parse model output extracting reasoning (analysis) and response (final)"""
    raw = self._tok.decode(ids, skip_special_tokens=False)
    
    # Clean all Harmony special tokens from extracted content
    def clean(s):
      if not s: return s
      for tok in (self.START, self.END, self.CHANNEL, self.MESSAGE, self.RETURN, "<|endoftext|>"):
        s = s.replace(tok, "")
      # Also clean any remaining special token patterns
      s = re.sub(r'<\|[^|]+\|>', '', s)
      # Clean trailing garbage text that looks like model confusion (e.g., "**analysis", "**user")
      s = re.sub(r'\*{0,2}(analysis|user|assistant|final|commentary|channel|message)\s*$', '', s, flags=re.IGNORECASE)
      return s.strip()
    
    # Extract analysis and final channels
    # Stop at any special token or end of string - special tokens in middle of content indicate model confusion
    stop_pattern = r'(?:<\|end\|>|<\|return\|>|<\|channel\|>|<\|start\|>|<\|message\|>|$)'
    analysis = [clean(m.group(1)) for m in re.finditer(rf'<\|channel\|>analysis<\|message\|>(.*?){stop_pattern}', raw, re.DOTALL) if m.group(1).strip()]
    final = [clean(m.group(1)) for m in re.finditer(rf'<\|channel\|>final<\|message\|>(.*?){stop_pattern}', raw, re.DOTALL) if m.group(1).strip()]
    
    # Fallback: extract after any channel or decode raw
    if not analysis and not final:
      m = re.search(r'<\|channel\|>[^<]*<\|(?:message|reserved_\d+)\|>(.*?)(?:<\|end\|>|<\|return\|>|$)', raw, re.DOTALL)
      if m: final = [clean(m.group(1))]
      elif '<|channel|>' not in raw: final = [self.decode(ids)]
    
    return "\n".join(analysis) or None, "\n".join(final) or None, raw

  @property
  def vocab_size(self) -> int: return self._tok.get_vocab_size()
