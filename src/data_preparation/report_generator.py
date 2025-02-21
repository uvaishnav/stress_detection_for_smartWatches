import pandas as pd
import plotly.express as px
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import json
from datetime import datetime
from jinja2 import TemplateNotFound

class ReportGenerator:
    def __init__(self, unified_df: pd.DataFrame, output_dir: Path):
        self.df = unified_df
        self.output_dir = output_dir
        
        # Debug: Print template search paths
        template_dir = Path('../src/templates').resolve()
        print(f"🛠️ [DEBUG] Looking for templates in: {template_dir}")
        print(f"🛠️ [DEBUG] Template exists? {template_dir.exists()}")
        
        self.env = Environment(loader=FileSystemLoader(str(template_dir)))
        self.report_data = {
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_records': len(unified_df),
            'sections': []
        }
        
    def generate_report(self):
        """Main method to generate full report"""
        self._add_summary_stats()
        self._add_data_quality()
        self._add_sensor_distributions()
        self._add_temporal_analysis()
        self._add_demographic_metadata()
        
        self._render_html()
        self._save_assets()
        
    def _add_summary_stats(self):
        """Add dataset summary statistics"""
        stats = {
            'total_subjects': self.df['subject_id'].nunique(),
            'total_hours': len(self.df) / (4 * 3600),  # 4Hz sampling
            'start_time': self.df.index.min().strftime('%Y-%m-%d %H:%M'),
            'end_time': self.df.index.max().strftime('%Y-%m-%d %H:%M'),
            'dataset_dist': self.df['dataset'].value_counts().to_dict(),
            'device_dist': self.df['device'].value_counts().to_dict()
        }
        
        self.report_data['sections'].append({
            'title': 'Dataset Summary',
            'type': 'stats',
            'content': stats
        })
    
    def _add_data_quality(self):
        """Add data quality metrics"""
        quality = {
            'missing_values': self.df.isna().mean().to_dict(),
            'out_of_range': {
                'bvp': ((self.df['bvp'] < -1) | (self.df['bvp'] > 1)).mean(),
                'acc': ((self.df[['acc_x','acc_y','acc_z']].abs() > 6).any(axis=1)).mean()
            }
        }
        
        self.report_data['sections'].append({
            'title': 'Data Quality Analysis',
            'type': 'quality',
            'content': quality
        })
        
    def _add_sensor_distributions(self):
        """Generate sensor distribution visualizations"""
        figs = {
            'bvp_dist': px.histogram(self.df, x='bvp', 
                                   title='BVP Distribution').to_html(),
            'acc_magnitude': px.histogram(self.df, x=(self.df[['acc_x','acc_y','acc_z']]**2).sum(axis=1)**0.5,
                                        title='Acceleration Magnitude Distribution').to_html()
        }
        
        self.report_data['sections'].append({
            'title': 'Sensor Distributions',
            'type': 'visualization',
            'content': figs
        })
        
    def _add_temporal_analysis(self):
        """Add time-related analysis"""
        # Resample and rename index column
        timeline = self.df.resample('1H').size().reset_index(name='count')
        timeline = timeline.rename(columns={'index': 'timestamp'})
        
        timeline_fig = px.line(timeline, x='timestamp', y='count', 
                             title='Data Timeline').to_html()
                             
        self.report_data['sections'].append({
            'title': 'Temporal Analysis',
            'type': 'visualization',
            'content': {'timeline': timeline_fig}
        })
        
    def _add_demographic_metadata(self):
        """Add demographic distributions"""
        skin_tone_fig = px.pie(self.df, names='skin_tone', 
                             title='Skin Tone Distribution').to_html()
        device_fig = px.pie(self.df, names='device', 
                          title='Device Distribution').to_html()
        
        self.report_data['sections'].append({
            'title': 'Demographic Metadata',
            'type': 'visualization',
            'content': {
                'skin_tone': skin_tone_fig,
                'devices': device_fig
            }
        })
        
    def _render_html(self):
        """Render HTML using Jinja template"""
        try:
            template = self.env.get_template('report_template.html')
        except TemplateNotFound as e:
            print(f"🔥 [ERROR] Template search paths: {self.env.loader.searchpath}")
            print(f"🔥 [ERROR] Files in template directory:")
            template_dir = Path(self.env.loader.searchpath[0])
            for f in template_dir.glob('*'):
                print(f" - {f.name}")
            raise

        html = template.render(report=self.report_data)
        
        report_path = self.output_dir / 'data_prep_report.html'
        with open(report_path, 'w') as f:
            f.write(html)
            
    def _save_assets(self):
        """Save CSS assets locally"""
        assets_dir = self.output_dir / 'assets'
        assets_dir.mkdir(exist_ok=True)
        
        # Update path to source CSS in templates/assets
        from shutil import copy
        copy('../src/templates/assets/style.css', assets_dir)