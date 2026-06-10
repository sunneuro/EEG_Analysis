import markdown2
import sys
import subprocess
import os

def main():
    with open('README.md', 'r', encoding='utf-8') as f:
        md = f.read()
    
    html = markdown2.markdown(md, extras=["tables", "fenced-code-blocks", "header-ids"])
    
    css = """
    <style>
      body {
        font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
        line-height: 1.6;
        padding: 40px;
        color: #24292e;
        max-width: 900px;
        margin: 0 auto;
      }
      h1, h2, h3, h4 { margin-top: 24px; margin-bottom: 16px; font-weight: 600; line-height: 1.25; }
      h1 { font-size: 2em; padding-bottom: .3em; border-bottom: 1px solid #eaecef; }
      h2 { font-size: 1.5em; padding-bottom: .3em; border-bottom: 1px solid #eaecef; }
      pre { padding: 16px; overflow: auto; line-height: 1.45; background-color: #f6f8fa; border-radius: 3px; }
      code { padding: .2em .4em; font-family: ui-monospace,SFMono-Regular,SF Mono,Menlo,Consolas,Liberation Mono,monospace; font-size: 85%; background-color: rgba(27,31,35,.05); border-radius: 3px; }
      pre code { padding: 0; background-color: transparent; }
      table { border-spacing: 0; border-collapse: collapse; margin-top: 0; margin-bottom: 16px; width: 100%; }
      table th, table td { padding: 6px 13px; border: 1px solid #dfe2e5; }
      table tr:nth-child(2n) { background-color: #f6f8fa; }
      blockquote { padding: 0 1em; color: #6a737d; border-left: .25em solid #dfe2e5; margin: 0; }
    </style>
    """
    
    final_html = f"<!DOCTYPE html>\n<html>\n<head>\n<meta charset='utf-8'>\n{css}\n</head>\n<body>\n{html}\n</body>\n</html>"
    with open('README.html', 'w', encoding='utf-8') as f:
        f.write(final_html)

    chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    html_path = f"file://{os.path.abspath('README.html')}"
    cmd = [chrome_path, "--headless", "--disable-gpu", "--print-to-pdf=README.pdf", "--no-pdf-header-footer", html_path]
    print(f"Running Chrome: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("Done generating README.pdf")

if __name__ == '__main__':
    main()
