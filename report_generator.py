import os
import re
from fpdf import FPDF
from datetime import datetime

class ReportGenerator:
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _clean_text(self, text, fallback="Analysis pending further data points."):
        if not text or str(text).strip() == "":
            return fallback
        if isinstance(text, list):
            text = "\n\n".join([str(t) for t in text])
        
        text = str(text)
        # Handle common encoding issues for PDF (Latin-1 compatible)
        replacements = {
            '\u2022': '-', '\u2013': '-', '\u2014': '-',
            '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
            '\u2192': '->', '\xb7': '*'
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text.encode('latin-1', 'replace').decode('latin-1')

    def _draw_header(self, pdf, title):
        pdf.set_font("Helvetica", 'B', 16)
        pdf.set_text_color(44, 62, 80) # Navy Blue
        pdf.cell(0, 10, title, ln=True)
        pdf.set_draw_color(44, 62, 80)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

    def generate_document(self, ml_results, ai_narrative, session_id, visual_description=None):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        visual_description = visual_description or {}
        
        # Data-Agnostic Title
        target = ml_results.get('target', 'Target Variable').replace('_', ' ').title()
        score = ml_results.get('score', 0)

        # --- PAGE 1: COVER ---
        pdf.add_page()
        pdf.set_fill_color(28, 40, 51) # Darker Professional Navy
        pdf.rect(0, 0, 210, 297, 'F')
        pdf.set_text_color(255, 255, 255)
        
        pdf.set_y(100)
        pdf.set_font("Helvetica", 'B', 32)
        pdf.cell(0, 15, "EXECUTIVE STRATEGY REPORT", ln=True, align='C')
        
        pdf.set_font("Helvetica", '', 16)
        pdf.ln(5)
        pdf.cell(0, 10, f"Optimization Analysis: {target}", ln=True, align='C')
        
        pdf.set_y(260)
        pdf.set_font("Helvetica", 'I', 10)
        pdf.cell(0, 10, f"Proprietary AI Analysis | Generated {datetime.now().strftime('%B %d, %Y')}", align='C')

        # --- PAGE 2: EXECUTIVE SUMMARY ---

        # --- EXECUTIVE SUMMARY (STRICT FLUSH LEFT) ---
        pdf.add_page()
        pdf.set_text_color(0, 0, 0)
        self._draw_header(pdf, "1. Executive Summary")
        
        summary_raw = ai_narrative.get("summary", "")
        
        # 1. SPLIT: Remove the pipe '|' and extra spaces during the split
        summary_points = [p.strip() for p in re.split(r'[|•\n]|(?<=[a-z0-9])\.\s', summary_raw) if len(p.strip()) > 10]
        
        for p in summary_points[:3]:
            # 2. CLEAN: Deep clean any remaining leading characters or spaces
            clean_p = re.sub(r'^[\s|•\-\d\.]+', '', p).strip()
            
            if clean_p:
                if not clean_p.endswith('.'):
                    clean_p += "."

                pdf.set_fill_color(245, 247, 250)
                pdf.set_font("Helvetica", '', 11)
                
                # 3. RENDER: Removed the leading space inside the f-string
                # align='L' and border=0 ensures it starts exactly at the left margin
                pdf.multi_cell(0, 10, self._clean_text(clean_p), fill=True, align='L', border=0)
                pdf.ln(2)

        # --- DATA & METHODOLOGY ---
        pdf.ln(3)
        self._draw_header(pdf, "2. Analytical Framework")
        methodology = f"{ai_narrative.get('data_logic', '')}\n\n{ai_narrative.get('methodology', '')}"
        pdf.set_font("Helvetica", '', 11)
        pdf.multi_cell(0, 7, self._clean_text(methodology))

        # --- MODEL PERFORMANCE (Data Agnostic) ---
        pdf.ln(3)
        self._draw_header(pdf, "3. Model Confidence & Performance")
        pdf.set_fill_color(240, 240, 240)
        pdf.set_font("Helvetica", 'B', 11)
        pdf.cell(95, 10, " Predictive Accuracy (R-Squared / Score)", 1, 0, 'L', fill=True)
        pdf.set_font("Helvetica", '', 11)
        pdf.cell(95, 10, f" {score:.4%}", 1, 1, 'L')
        
        # --- KEY DRIVERS (Feature Importance) ---
        pdf.add_page()
        self._draw_header(pdf, "4. Critical Success Drivers")
        pdf.set_font("Helvetica", 'I', 10)
        pdf.multi_cell(0, 6, f"The following variables demonstrate the highest statistical impact on {target}:")
        pdf.ln(4)

        fi = ml_results.get("feature_importance_dict", {})
        if fi:
            # Scale-agnostic sorting
            sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (feat, val) in enumerate(sorted_fi, 1):
                pdf.set_fill_color(255, 255, 255)
                # Visual bar representation
                bar_width = min(val * 100, 100) if val <= 1 else val # Handle different importance scales
                pdf.set_font("Helvetica", 'B', 10)
                pdf.cell(50, 8, f"{feat.replace('_', ' ')}", 0)
                pdf.set_fill_color(44, 62, 80)
                pdf.cell(bar_width, 8, "", 0, 0, 'L', fill=True)
                pdf.set_font("Helvetica", '', 9)
                pdf.cell(20, 8, f"  {val:.3f}", 0, 1)
                pdf.ln(3)
        
        # --- BUSINESS IMPACT & ROI ---
        pdf.ln(2)
        self._draw_header(pdf, "5. Estimated Business Impact")
        pdf.ln(2) 
        
        impact_text = ai_narrative.get("business_impact", "")
        
        # 1. THE SPLITTER: This regex catches 1., (1), or newlines to find each point
        impact_points = [p.strip() for p in re.split(r'\d+\.|\(\d+\)|[•\n]', impact_text) if p and len(p.strip()) > 5]

        # 2. THE RENDERER
        for i, point in enumerate(impact_points[:7], 1):
            # Clean the text to remove leading dots, colons, or spaces
            clean_point = re.sub(r'^[:\-\s\d\.)]+', '', point).strip()
            
            if clean_point:
                # Ensure the point ends with a period
                if not clean_point.endswith('.'):
                    clean_point += "."

                pdf.set_font("Helvetica", '', 11)
                pdf.set_text_color(0, 0, 0)
                
                # Format: "1. Point text..."
                numbered_line = f"{i}. {self._clean_text(clean_point)}"
                
                # 3. FORCE LINE BREAK: multi_cell(0, ...) uses the full width of the page
                # and automatically handles text wrapping for long sentences.
                pdf.multi_cell(0, 8, numbered_line, border=0, align='L')
                
                # 4. VERTICAL GAP: This forces the NEXT point (e.g., 2.) to jump 
                # down 5mm, ensuring they never look like a paragraph.
                pdf.ln(5)

        # Strategic ROI Projection Block
        pdf.ln(2)
        pdf.set_fill_color(232, 245, 233) 
        pdf.set_font("Helvetica", 'B', 12)
        pdf.cell(0, 10, " Strategic ROI Projection", ln=True, fill=True)
        pdf.set_font("Helvetica", '', 11)
        pdf.multi_cell(0, 8, self._clean_text(ai_narrative.get("roi_simulation", "Analysis pending.")), fill=True)

        # --- RISK & KPI DASHBOARD ---
        pdf.add_page()
        self._draw_header(pdf, "6. Risk Mitigation & KPIs")
        
        # 1. Risk Assessment (Cleaned up)
        pdf.set_font("Helvetica", 'B', 12)
        pdf.set_text_color(176, 58, 46) 
        pdf.cell(0, 10, "Strategic Risk Assessment", ln=True)
        pdf.set_font("Helvetica", '', 11)
        pdf.set_text_color(0, 0, 0)
        pdf.set_fill_color(254, 235, 238) 
        pdf.multi_cell(0, 8, self._clean_text(ai_narrative.get("risk_analysis")), fill=True)
        
        pdf.ln(10)
        
        # 2. KPI Dashboard (Fixed: No brackets, No trailing line)
    
        pdf.set_font("Helvetica", 'B', 12)
        pdf.set_text_color(44, 62, 80) 
        pdf.cell(0, 10, "Success Measurement Dashboard (KPIs)", ln=True)
        pdf.ln(2) # Space after title

        raw_kpis = ai_narrative.get("kpi_dashboard", "")
        
        # 1. THE SPLITTER: Captures (1), (2), etc., even if the AI mashes them together
        parts = re.split(r'(\(\d+\))', raw_kpis)
        
        kpi_list = []
        for i in range(1, len(parts), 2):
            number = parts[i]
            # Clean the text: remove leading dots, spaces, or dashes
            text = parts[i+1].strip() if i+1 < len(parts) else ""
            text = re.sub(r'^[.\s:-]+', '', text) 
            kpi_list.append(f"{number} {text}")

        # 2. THE RENDERER: Force separate lines with NO borders
        for clean_kpi in kpi_list[:7]:
            pdf.set_font("Helvetica", '', 11)
            pdf.set_text_color(0, 0, 0)
            
            # We set border=0 to remove all lines in between
            # fill=False keeps the background clean white like your example
            pdf.multi_cell(0, 8, self._clean_text(clean_kpi), border=0, align='L')
            
            # 3. This 'ln' ensures the next point starts on a brand new line
            pdf.ln(2)

        # --- VISUALS (BOARDROOM STRUCTURED) ---

        if os.path.exists("visuals"):
            # Ensure images are processed in a consistent order
            imgs = sorted([f for f in os.listdir("visuals") if f.endswith((".png", ".jpg"))])
            
            for img in imgs:
                pdf.add_page()
                # Create a clean title from the filename
                title = img.replace(".png", "").replace(".jpg", "").replace("_", " ").title()
                self._draw_header(pdf, f"Visual Analysis: {title}")
                
                # Insert the Chart
                img_path = os.path.join("visuals", img)
                pdf.image(img_path, x=15, w=180)
                pdf.ln(5)
                
                # --- DIRECT FILENAME LOOKUP ---
                # This now matches the os.path.basename key we set in visualization.py
                desc = visual_description.get(img, "Analysis pending.")
                
                # Clean the AI text: remove asterisks and extra whitespace
                clean_desc = self._clean_text(desc).replace("**", "")
                
                # Split the description into structural blocks
                sentences = [s.strip() for s in re.split(r'\. |\n', clean_desc) if len(s.strip()) > 5]
                
                # Fallback logic if the AI description is short
                key_insight = sentences[0] if len(sentences) > 0 else clean_desc
                why_matters = ". ".join(sentences[1:3]) if len(sentences) > 2 else "Patterns suggest a direct correlation with target performance metrics."
                so_what = ". ".join(sentences[3:]) if len(sentences) > 3 else "Prioritize strategic resources based on this variance."

                # Internal helper for the Insight Blocks
                def draw_styled_block(label, text, header_color, bg_color):
                    pdf.set_fill_color(*header_color)
                    pdf.set_text_color(255, 255, 255)
                    pdf.set_font("Helvetica", 'B', 11)
                    pdf.cell(0, 8, f"  {label}", ln=True, fill=True)
                    
                    pdf.set_fill_color(*bg_color)
                    pdf.set_text_color(0, 0, 0)
                    pdf.set_font("Helvetica", '', 10)
                    # The multi_cell handles line wrapping within the colored block
                    pdf.multi_cell(0, 7, f" {text}", fill=True)
                    pdf.ln(2)

                # Render the Boardroom Insight Blocks
                draw_styled_block("KEY INSIGHT", key_insight, (44, 62, 80), (235, 245, 255))
                draw_styled_block("WHY IT MATTERS", why_matters, (39, 174, 96), (235, 255, 240))
                draw_styled_block("SO WHAT / ACTION", so_what, (192, 57, 43), (255, 240, 240))

        # --- STRATEGIC ROADMAP ---
        
        pdf.add_page()
        self._draw_header(pdf, "7. Strategic Roadmap")
        pdf.ln(5)

        roadmap_text = ai_narrative.get('roadmap', '')
        
        # 1. THE SPLITTER: Captures (1), Step 1, or newlines to get individual actions
        # This makes it dataset-agnostic by finding the intent of the list
        roadmap_pts = [p.strip() for p in re.split(r'\(\d+\)|\d+\.|Step\s*\d+', roadmap_text) if len(p.strip()) > 5]

        # 2. THE RENDERER: Clean, flush-left numbering with no heavy blocks
        for i, point in enumerate(roadmap_pts, 1):
            # CLEANING: Strip any leftover prefix characters
            clean_point = re.sub(r'^[:\-\s\d\.)]+', '', point).strip()
            
            # Sub-Header for the number: (1), (2), etc.
            pdf.set_font("Helvetica", 'B', 11)
            pdf.set_text_color(44, 62, 80) # Professional Navy
            numbering = f"({i})"
            
            # Print the number
            pdf.cell(10, 8, numbering, ln=0)
            
            # Print the Action Text
            pdf.set_font("Helvetica", '', 11)
            pdf.set_text_color(0, 0, 0)
            
            # multi_cell ensures the text wraps perfectly if the action is long
            # Using a small left margin (15) to keep text aligned next to the number
            current_x = pdf.get_x()
            pdf.multi_cell(0, 8, self._clean_text(clean_point), border=0, align='L')
            
            # 3. Add a vertical gap between roadmap items for "breathing room"
            pdf.ln(5)

        # Footer note for the Roadmap
        pdf.set_y(-30)
        pdf.set_font("Helvetica", 'B', 12)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 10, "Timeline: Immediate implementation recommended for Phase 1 objectives.", align='C')

        # Save
        output_file = os.path.join(self.output_dir, f"Executive_Report_{session_id}.pdf")
        pdf.output(output_file)
        return output_file