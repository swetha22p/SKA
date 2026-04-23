import json
import os
from pathlib import PurePath
from wxconv import WXC

class JsonFormatter:
    def __init__(self, input_folder, output_folder, log_folder="usr_error_logs"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.log_folder = log_folder
        self.wxc = WXC(order='wx2utf', lang='hin')

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

    @staticmethod
    def extract_usr_blocks_from_file(file_path):
        with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
            content = f.read()

        content = content.replace("\r\n", "\n").strip()
        blocks = []
        current_block = []
        inside_block = False

        for line in content.splitlines():
            line = line.strip()
            # Added on 30-July-2025 to remove the invisible BOM marker character from start of a line
            if line.startswith('\ufeff'):
                line = line[1:]
            if line.startswith("<sent_id") or line.startswith("<segment_id"):
                inside_block = True
                current_block = [line]
            elif line.startswith("</sent_id>") or line.startswith("</segment_id>") and inside_block:
                current_block.append(line)
                blocks.append("\n".join(current_block))
                inside_block = False
                current_block = []
            elif inside_block:
                current_block.append(line)

        return blocks

    @staticmethod
    def clear_error_log(filepath="error.txt"):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("USR Parsing Errors Log\n")

    @staticmethod
    def log_failed_usr_blocks(failed_ids, log_path="error.txt"):
        if not failed_ids:
            print("\nAll USR blocks parsed successfully.")
            return

        print("Skipped malformed USR blocks:")
        with open(log_path, "a", encoding="utf-8") as f:
            for sent_id, error in failed_ids:
                print(f" - {sent_id} | {error}")
                f.write(f"{sent_id} - {error}\n")

    def parse_usr_block_to_json(self, block, failed_ids, seen_ids):
        lines = block.strip().splitlines()
        sentence = {"text": None, "usr_id": None, "SENT_TYPE": None, "tokens": []}
        malformed = False
        sent_id = None
        error = None
        token_lines = []

        for line in lines:
            line = line.strip()

            if line.startswith("<sent_id=") or line.startswith("<segment_id"):
                sent_id = line.split("=")[1].strip(">").strip()
                if sent_id in seen_ids:
                    error = f"Duplicate usr_id: {sent_id}"
                    malformed = True
                    break

                seen_ids.add(sent_id)
                sentence["usr_id"] = sent_id

            elif line.startswith("#"):
                text = " ".join(line.split("#")[1:]).strip()
                sentence["text"] = text

            elif line.startswith("%"):
                sentence["SENT_TYPE"] = line.strip("%").strip()

            elif line.startswith("</sent_id") or line.startswith("</segment_id>"):
                continue

            elif not any(line.startswith(prefix) for prefix in ("<", "#", "%")):
                if line.strip():
                    token_lines.append(line)

        if not token_lines and not malformed:
            error = "Empty USR block (no tokens)"
            malformed = True

        if not malformed and sentence.get("SENT_TYPE") is None:
            error = "Missing sent_type"
            malformed = True

        if malformed:
            if sent_id:
                failed_ids.append([sent_id, error])
            return None

        for line in token_lines:
            parts = line.split()

            if len(parts) != 9:
                error = f"Incorrect number of columns: {len(parts)} (expected 9)"
                malformed = True
                break

            concept_raw = parts[0]
            try:
                token = {
                    "index": int(parts[1]),
                }
            except ValueError:
                error = f"Invalid token index: {parts[1]}"
                malformed = True
                break

            sem_cat = parts[2]
            morpho = parts[3]
            dep_info = parts[4]
            discourse_info = parts[5]
            speaker_view = parts[6]
            scope_info = parts[7]
            construct_info = parts[8]

            if concept_raw.startswith("[") and concept_raw.endswith("]"):
                concept_base = concept_raw
            elif '-' in concept_raw:
                concept_parts = concept_raw.split('-', 1)
                if len(concept_parts) != 2 or not concept_parts[1]:
                    error = f"Malformed TAM in concept: {concept_raw}"
                    malformed = True
                    break
                concept_base = concept_parts[0]
                tam = concept_parts[1]
                try:
                    if tam in ["pres", "past"]:
                        token["tam"] = tam
                    else:
                        tam_parts = tam.split("_")
                        converted_tam = "_".join(self.wxc.convert(p) if not p.isdigit() else p for p in tam_parts)
                        token["tam"] = converted_tam
                    token["is_combined_tam"] = True
                except Exception as e:
                    error = f"Error parsing TAM in concept: {concept_raw}, {e}"
                    malformed = True
                    break
            else:
                concept_base = concept_raw

            if (
                not concept_base.startswith('[')
                and not concept_base.startswith('$')
                and not concept_base.startswith('^')
                and not concept_base.startswith('@')
            ):
                if "_" in concept_base:
                    concept_text, suffix = concept_base.split("_", 1)
                    converted = self.wxc.convert(concept_text)
                    token["concept"] = f"{converted}_{suffix}"
                else:
                    token["concept"] = self.wxc.convert(concept_base)
            else:
                token["concept"] = concept_base

            if sem_cat != "-":
                token["sem_category"] = sem_cat
            if morpho != "-":
                token["morpho_sem"] = morpho

            if dep_info != "-":
                if ":" not in dep_info:
                    error = f"Malformed dep_info: {dep_info}"
                    malformed = True
                    break
                head, rel = dep_info.split(":", 1)
                token["dep_rel"] = rel
                token["dep_head"] = int(head) if head.isdigit() else head

            if discourse_info != "-":
                if ":" not in discourse_info:
                    error = f"Malformed discourse_info: {discourse_info}"
                    malformed = True
                    break
                dh, dr = discourse_info.split(":", 1)
                token["discourse_head"] = dh
                token["discourse_rel"] = dr

            if speaker_view != "-":
                token["speaker_view"] = speaker_view

            if construct_info != "-":
                if ":" not in construct_info:
                    error = f"Malformed cxn_info: {construct_info}"
                    malformed = True
                    break
                if dep_info != "-":
                    error = "Cannot have both dep_info and cxn_info"
                    malformed = True
                    break
                ch, tag = construct_info.split(":", 1)
                try:
                    token["cxn_head"] = int(ch)
                except ValueError:
                    token["cxn_head"] = float(ch)

                token["cxn_tag"] = tag

            sentence["tokens"].append(token)

        if malformed:
            if sent_id:
                failed_ids.append([sent_id, error])
            return None

        return sentence

    @staticmethod
    def resolve_index_to_concept(ref, all_graphs):
        if not isinstance(ref, str) or "." not in ref:
            return None

        try:
            sent_id, idx_str = ref.rsplit(".", 1)
            idx = int(idx_str)
            graph = all_graphs.get(sent_id)

            if not graph:
                return None

            for node in graph.get("tokens", []):
                if node.get("index") == idx:
                    return node.get("concept")
            return None
        except Exception as e:
            print(f"Failed to resolve concept for {ref}: {e}")
            return None

    def json_to_graph(self, usr_data, all_json, failed_ids):
        tokens = usr_data["tokens"]
        usr_id = usr_data.get("usr_id", "").strip()
        nodes = []
        edges_dep = []
        edges_cxn = []
        edges_discourse = []

        disc_part_values = {
            "hI_1", "hI_2", "hI_3", "hI_4", "hI_5", "hI_6",
            "BI_1", "BI_2", "BI_3", "BI_4", "BI_5",
            "wo_1", "kevala_1", "sirPa_1", "lagaBaga_1", "sA_1",
            "hAz_1", "karIba_1", "sA_1", "nA_1", "sI_1", "mAwra_1",
            "ki_1", "waka_1"
        }

        shade_values = {
            "jA_1", "jA_2", "dAla_1", "dAla_2", "ho_1",
            "pA_1", "xe_1", "le_1", "uTa_1", "bETa_1",
            "Cala_1", "laga_1", "mara_1", "A_1", "bana_1"
        }

        deixis_field_values = {"proximal", "distal"}

        idx2concept = {}

        for token in tokens:
            index = token.get("index")
            concept = token.get("concept")

            node = {
                "index": index,
                "concept": concept
            }

            full_ref = f"{usr_id}.{index}/{concept}"
            idx2concept[index] = full_ref

            properties = {}

            structural_keys = {
                "index", "concept", "dep_rel", "dep_head",
                "discourse_rel", "discourse_head",
                "cxn_tag", "cxn_head"
            }

            for key, value in token.items():
                if key in structural_keys:
                    continue

                if key == "sem_category" and isinstance(value, str):
                    for part in value.split("/"):
                        if part in ["male", "female"]:
                            properties["attr_gen"] = part
                        elif part in ["per", "place", "org"]:
                            properties["attr_enamex"] = part
                        elif part in ["yoc", "moy", "season", "dow", "dom",
                                      "calendricunit", "clocktime", "timex", "era"]:
                            properties["attr_timex"] = part
                        elif part == "ne":
                            properties["attr_ne"] = part
                        elif part == "numex":
                            properties["attr_num"] = part
                        elif part == "anim":
                            properties["attr_animacy"] = part

                elif key == "morpho_sem" and isinstance(value, str):
                    if "pl" in value:
                        properties["attr_num"] = "pl"
                    if "mawup" in value:
                        properties["attr_mawubarWa"] = "mawup"
                    if "doublecausative" in value:
                        properties["attr_causative"] = "doublecausative"
                    elif "causative" in value:
                        properties["attr_causative"] = "causative"
                    if "comparmore" in value:
                        properties["attr_comparison"] = "comparmore"
                    elif "comparless" in value:
                        properties["attr_comparison"] = "comparless"
                    elif "superl" in value:
                        properties["attr_comparison"] = "superl"
                    if "dviwva" in value:
                        properties["attr_redup"] = "dviwva"

                elif key == "speaker_view":
                    if value in disc_part_values:
                        properties["disc_part"] = value
                    elif value in shade_values:
                        properties["shade"] = value
                    elif value == "respect":
                        properties["respect"] = "yes"
                    elif value == "informal":
                        properties["informal"] = "yes"
                    elif value == "def":
                        properties["definite"] = "yes"
                    elif value in deixis_field_values:
                        properties["deixis"] = value
                    else:
                        properties["speaker_view"] = value

                else:
                    properties[key] = value

            if token.get("dep_head") == 0:
                properties["is_root"] = True

            if properties:
                node["properties"] = properties

            if "cxn_tag" in token and "cxn_head" in token:
                node["components"] = {
                    "cxn_tag": token["cxn_tag"],
                    "cxn_head_index": token["cxn_head"]  # temporarily store index
                }

            nodes.append(node)

        for node in nodes:
            if "components" in node and "cxn_head_index" in node["components"]:
                head_index = node["components"].pop("cxn_head_index")
                cxn_head_concept = idx2concept.get(head_index)
                node["components"]["cxn_head"] = cxn_head_concept

                tail_index = node.get("index")
                tail_ref = f"{usr_id}.{tail_index}/{node['concept']}" if tail_index is not None else None

                if cxn_head_concept and tail_ref:
                    edges_cxn.append([cxn_head_concept, node["components"]["cxn_tag"], tail_ref])

                node.pop("components", None)

        for token in tokens:
            rel = token.get("dep_rel")
            head_idx = token.get("dep_head")
            if head_idx is None:
                continue

            if rel in ["-", "main"] or head_idx in ["-", 0]:
                continue

            if isinstance(token["dep_head"], int):
                head_concept = idx2concept.get(token.get("dep_head"))
            else:
                concept_value = self.resolve_index_to_concept(token.get("dep_head"), all_json)
                head_concept = f'{token.get("dep_head")}/{concept_value}'

            tail_concept = idx2concept.get(token.get("index")) # token["concept"]
            if head_concept and tail_concept:
                edges_dep.append([head_concept, rel, tail_concept])

        for token in tokens:
            if "discourse_head" in token and "discourse_rel" in token:
                discourse_rel = token.get("discourse_rel")
                discourse_head = token.get("discourse_head")

                if discourse_rel and discourse_head is not None:
                    tail_ref = idx2concept.get(token['index'])

                    if isinstance(discourse_head, int) or (isinstance(discourse_head, str) and discourse_head.isdigit()):
                        idx = int(discourse_head)
                        head_concept = idx2concept.get(idx)
                        head_ref = f"{head_concept}"
                    else:
                        try:
                            sent_id, idx_str = discourse_head.rsplit(".", 1)
                            idx = int(idx_str)
                            head_graph = all_json.get(sent_id)
                        except Exception:
                            failed_ids.append([usr_id, "Malformed discourse head"])
                            break

                        if not head_graph:
                            continue

                        head_concept = None
                        for node in head_graph.get("tokens", []):
                            if node.get("index") == idx:
                                head_concept = node.get("concept")
                                break

                        if not head_concept:
                            continue

                        head_ref = f"{sent_id}.{idx}/{head_concept}"

                    edges_discourse.append([head_ref, discourse_rel, tail_ref])

        return {
            "text": usr_data["text"],
            "usr_id": usr_id,
            "SENT_TYPE": usr_data.get("SENT_TYPE", ""),
            "nodes": nodes,
            "edges_dep": edges_dep,
            "edges_cxn": edges_cxn,
            "edges_discourse": edges_discourse
        }

    def process(self):
        for filename in os.listdir(self.input_folder):
            if not filename.endswith(".txt"):
                continue
            input_file_path = os.path.join(self.input_folder, filename)

            output_filename = PurePath(filename).stem
            output_filename = f"{output_filename}.json"
            output_file_path = os.path.join(
                self.output_folder, output_filename
            )

            usr_blocks = self.extract_usr_blocks_from_file(input_file_path)

            error_filename = f"error-{PurePath(filename).stem}.txt"
            error_file_path = os.path.join(self.log_folder, error_filename)

            seen_ids = set()
            failed_ids = []
            converted_jsons = []

            for block in usr_blocks:
                result = self.parse_usr_block_to_json(block, failed_ids, seen_ids)
                if result:
                    converted_jsons.append(result)

            all_graphs_dict = {g.get("usr_id"): g for g in converted_jsons}

            transformed = [self.json_to_graph(entry, all_graphs_dict, failed_ids) for entry in converted_jsons]

            if failed_ids:
                self.clear_error_log(error_file_path)
                self.log_failed_usr_blocks(failed_ids, error_file_path)
                print(f"Logged {len(failed_ids)} errors to {error_file_path}")
            else:
                print(f"No errors for {filename}")

            json.dump(transformed, open(output_file_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
            print(f"\nProcessed {filename} and saved to {output_file_path}")

if __name__ == "__main__":
    # input_folder = r"InputDataSanskrit/usr"
    # output_folder = r"InputDataSanskrit/json"
    input_folder = r"bhagvatgeeta_input"
    output_folder = r"error_checking/json"
    log_folder = r"error_checking/usr_error_logs"

    json_formatter = JsonFormatter(
        input_folder=input_folder,
        output_folder=output_folder,
        log_folder=log_folder
    )

    json_formatter.process()