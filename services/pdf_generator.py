from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black, blue, darkblue
import markdown
import re
from datetime import datetime
import os

class PDFGenerator:
    """Generate PDF reports from markdown content."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=darkblue,
            alignment=1  # Center alignment
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=blue
        ))
        
        # Subheading style
        self.styles.add(ParagraphStyle(
            name='CustomSubheading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=8,
            textColor=darkblue
        ))
    
    def markdown_to_pdf(self, markdown_content: str, output_path: str, title: str = "Literature Review"):
        """Convert markdown content to PDF."""
        
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build story (content)
        story = []
        
        # Add title
        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Spacer(1, 20))
        
        # Add generation timestamp
        timestamp = f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
        story.append(Paragraph(timestamp, self.styles['Normal']))
        story.append(Spacer(1, 30))
        
        # Process markdown content
        lines = markdown_content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                story.append(Spacer(1, 6))
                continue
            
            # Handle headers
            if line.startswith('# '):
                # Main title (skip, already added)
                continue
            elif line.startswith('## '):
                # Section headers
                header_text = line[3:].strip()
                story.append(Paragraph(header_text, self.styles['CustomHeading']))
            elif line.startswith('### '):
                # Subsection headers
                header_text = line[4:].strip()
                story.append(Paragraph(header_text, self.styles['CustomSubheading']))
            else:
                # Regular paragraphs
                if line and not line.startswith('Generated:'):
                    # Clean up markdown formatting
                    clean_line = self._clean_markdown(line)
                    story.append(Paragraph(clean_line, self.styles['Normal']))
                    story.append(Spacer(1, 6))
        
        # Build PDF
        doc.build(story)
        return output_path
    
    def _clean_markdown(self, text: str) -> str:
        """Clean markdown formatting for PDF."""
        # Convert bold **text** to <b>text</b>
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        
        # Convert italic *text* to <i>text</i>
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        
        # Convert citations [Text, Year] to italic
        text = re.sub(r'\[(.*?)\]', r'<i>[\1]</i>', text)
        
        # Escape XML characters
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        
        # Restore our formatting tags
        text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
        text = text.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')
        
        return text

def generate_pdf_report(markdown_content: str, report_id: str, title: str = "Literature Review") -> str:
    """Generate PDF from markdown content."""
    
    # Ensure reports directory exists
    os.makedirs("reports", exist_ok=True)
    
    # Generate PDF
    pdf_path = f"reports/{report_id}.pdf"
    generator = PDFGenerator()
    generator.markdown_to_pdf(markdown_content, pdf_path, title)
    
    return pdf_path