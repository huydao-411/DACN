"""
Financial News Sentiment Analysis using FinBERT
================================================
Script phân tích Sentiment cho tin tức tài chính từ CafeF
sử dụng pre-trained model FinBERT từ Hugging Face

Features:
- Sử dụng ProsusAI/finbert (chuyên tài chính)
- Batch processing để tối ưu hiệu năng
- Cache model tránh load lại
- Xử lý text dài
- Tính sentiment score trung bình
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class FinBERTSentimentAnalyzer:
    """
    Class phân tích sentiment sử dụng FinBERT
    """
    
    # Class variable để cache model (tránh load lại)
    _model_cache = {}
    _tokenizer_cache = {}
    
    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = None):
        """
        Khởi tạo Sentiment Analyzer
        
        Args:
            model_name (str): Tên model từ Hugging Face
            device (str): 'cuda' hoặc 'cpu' (auto-detect nếu None)
        """
        self.model_name = model_name
        
        # Tự động detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Model: {model_name}")
        
        # Load model từ cache hoặc download
        self._load_model()
        
        # Label mapping
        self.label2sentiment = {
            0: 'positive',
            1: 'negative',
            2: 'neutral'
        }
        
        self.sentiment2score = {
            'positive': 1,
            'negative': -1,
            'neutral': 0
        }
    
    def _load_model(self):
        """
        Load model và tokenizer từ cache hoặc download
        """
        try:
            print(f"[INFO] Đang load model '{self.model_name}'...")
            
            # Kiểm tra cache
            if self.model_name not in self._model_cache:
                print("[INFO] Model không trong cache, đang download...")
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._tokenizer_cache[self.model_name] = tokenizer
                
                # Load model
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=3
                )
                model = model.to(self.device)
                model.eval()
                self._model_cache[self.model_name] = model
                
                print("[SUCCESS] Model downloaded và cached")
            else:
                print("[INFO] Model tải từ cache")
            
            self.model = self._model_cache[self.model_name]
            self.tokenizer = self._tokenizer_cache[self.model_name]
            
        except Exception as e:
            print(f"[ERROR] Lỗi khi load model: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str, max_length: int = 512) -> str:
        """
        Tiền xử lý text
        
        Args:
            text (str): Text gốc
            max_length (int): Độ dài tối đa
        
        Returns:
            str: Text đã xử lý
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Loại bỏ khoảng trắng thừa
        text = text.strip()
        
        # Nếu text quá dài, cắt bớt
        if len(text) > max_length:
            print(f"[WARNING] Text quá dài ({len(text)} chars), cắt bớt đến {max_length}")
            text = text[:max_length]
        
        return text
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Phân tích sentiment của một text
        
        Args:
            text (str): Text cần phân tích
        
        Returns:
            Tuple[str, float]: (label, confidence_score)
        """
        try:
            # Tiền xử lý
            text = self._preprocess_text(text)
            
            if not text:
                print("[WARNING] Text trống, trả về neutral")
                return 'neutral', 0.0
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Lấy probabilities
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            # Convert to Python types
            label = self.label2sentiment[predicted_class.item()]
            confidence = confidence.item()
            
            return label, confidence
        
        except Exception as e:
            print(f"[ERROR] Lỗi khi analyze: {str(e)}")
            return 'neutral', 0.0
    
    def analyze_batch(self, texts: List[str], batch_size: int = 8) -> List[Tuple[str, float]]:
        """
        Phân tích sentiment cho batch texts
        
        Args:
            texts (List[str]): Danh sách texts
            batch_size (int): Kích thước batch
        
        Returns:
            List[Tuple[str, float]]: Danh sách (label, confidence)
        """
        results = []
        total = len(texts)
        
        print(f"\n[INFO] Đang phân tích {total} texts trong batch...")
        
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = []
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                
                # Process results
                probabilities = torch.softmax(logits, dim=-1)
                confidences, predicted_classes = torch.max(probabilities, 1)
                
                for j in range(len(batch_texts)):
                    label = self.label2sentiment[predicted_classes[j].item()]
                    confidence = confidences[j].item()
                    batch_results.append((label, confidence))
                
                results.extend(batch_results)
                
                # Progress
                progress = min(i + batch_size, total)
                print(f"[PROGRESS] {progress}/{total} texts processed...")
                
            except Exception as e:
                print(f"[ERROR] Lỗi batch {i//batch_size}: {str(e)}")
                # Fallback: process individually
                for text in batch_texts:
                    label, conf = self.analyze_sentiment(text)
                    batch_results.append((label, conf))
                results.extend(batch_results)
        
        print(f"[SUCCESS] Phân tích hoàn thành {len(results)} texts\n")
        return results
    
    def calculate_sentiment_score(self, label: str, confidence: float) -> float:
        """
        Tính sentiment score từ label và confidence
        
        Args:
            label (str): Sentiment label
            confidence (float): Confidence score
        
        Returns:
            float: Sentiment score (-1 to 1)
        """
        base_score = self.sentiment2score.get(label, 0)
        weighted_score = base_score * confidence
        return weighted_score


def load_news_data(csv_file: str = "cafef_news.csv") -> pd.DataFrame:
    """
    Tải dữ liệu tin tức từ CSV
    
    Args:
        csv_file (str): Tên file CSV
    
    Returns:
        pd.DataFrame: DataFrame chứa tin tức
    """
    try:
        print(f"[INFO] Đang tải {csv_file}...")
        df = pd.read_csv(csv_file)
        print(f"[SUCCESS] Tải {len(df)} bài viết")
        return df
    except Exception as e:
        print(f"[ERROR] Lỗi tải file: {str(e)}")
        return pd.DataFrame()


def analyze_news_sentiment(df: pd.DataFrame, analyzer: FinBERTSentimentAnalyzer) -> pd.DataFrame:
    """
    Phân tích sentiment cho toàn bộ tin tức
    
    Args:
        df (pd.DataFrame): DataFrame tin tức
        analyzer (FinBERTSentimentAnalyzer): Analyzer object
    
    Returns:
        pd.DataFrame: DataFrame với sentiment columns
    """
    try:
        print("\n" + "="*80)
        print("PHÂN TÍCH SENTIMENT TIN TỨC")
        print("="*80)
        
        # Chuẩn bị dữ liệu
        # Sử dụng title + summary để có enough context
        texts = []
        for idx, row in df.iterrows():
            title = str(row.get('title', '')).strip()
            summary = str(row.get('summary', '')).strip()
            combined_text = f"{title}. {summary}"
            texts.append(combined_text)
        
        # Phân tích batch
        results = analyzer.analyze_batch(texts, batch_size=8)
        
        # Thêm vào DataFrame
        sentiment_labels = [label for label, conf in results]
        sentiment_confidences = [conf for label, conf in results]
        sentiment_scores = [
            analyzer.calculate_sentiment_score(label, conf)
            for label, conf in results
        ]
        
        df['sentiment_label'] = sentiment_labels
        df['sentiment_confidence'] = sentiment_confidences
        df['sentiment_score'] = sentiment_scores
        
        print("\n[SUCCESS] Phân tích sentiment hoàn thành")
        return df
    
    except Exception as e:
        print(f"[ERROR] Lỗi khi phân tích: {str(e)}")
        return df


def display_sentiment_stats(df: pd.DataFrame):
    """
    Hiển thị thống kê sentiment
    
    Args:
        df (pd.DataFrame): DataFrame với sentiment columns
    """
    try:
        print("\n" + "="*80)
        print("THỐNG KÊ SENTIMENT")
        print("="*80)
        
        # Phân bố sentiment
        print("\n[Phân bố Sentiment Labels]")
        sentiment_dist = df['sentiment_label'].value_counts()
        for label, count in sentiment_dist.items():
            percentage = (count / len(df)) * 100
            print(f"  {label:10}: {count:3} ({percentage:5.1f}%)")
        
        # Thống kê score
        print("\n[Thống kê Sentiment Score]")
        print(f"  Mean:     {df['sentiment_score'].mean():7.4f}")
        print(f"  Std:      {df['sentiment_score'].std():7.4f}")
        print(f"  Min:      {df['sentiment_score'].min():7.4f}")
        print(f"  Max:      {df['sentiment_score'].max():7.4f}")
        print(f"  Median:   {df['sentiment_score'].median():7.4f}")
        
        # Confidence
        print("\n[Thống kê Confidence]")
        print(f"  Mean:     {df['sentiment_confidence'].mean():7.4f}")
        print(f"  Min:      {df['sentiment_confidence'].min():7.4f}")
        print(f"  Max:      {df['sentiment_confidence'].max():7.4f}")
        
        # Overall Sentiment Score
        weighted_score = df['sentiment_score'].mean()
        print(f"\n[Overall Sentiment Score]: {weighted_score:7.4f}")
        if weighted_score > 0.1:
            print("  ➜ Trend: POSITIVE 📈")
        elif weighted_score < -0.1:
            print("  ➜ Trend: NEGATIVE 📉")
        else:
            print("  ➜ Trend: NEUTRAL ➡️")
        
    except Exception as e:
        print(f"[ERROR] Lỗi khi hiển thị stats: {str(e)}")


def display_sample_results(df: pd.DataFrame, num_samples: int = 5):
    """
    Hiển thị mẫu kết quả phân tích
    
    Args:
        df (pd.DataFrame): DataFrame
        num_samples (int): Số mẫu
    """
    try:
        print("\n" + "="*80)
        print(f"MẪU {num_samples} BÀI VIẾT VỚI SENTIMENT")
        print("="*80)
        
        for idx, (i, row) in enumerate(df.head(num_samples).iterrows(), 1):
            title = str(row.get('title', 'N/A'))[:60]
            label = row.get('sentiment_label', 'N/A')
            conf = row.get('sentiment_confidence', 0)
            score = row.get('sentiment_score', 0)
            
            emoji = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}.get(label, '❓')
            
            print(f"\n[{idx}] {title}...")
            print(f"    Label: {label:10} {emoji}")
            print(f"    Confidence: {conf:.4f}")
            print(f"    Score: {score:7.4f}")
        
        print("\n" + "="*80)
    
    except Exception as e:
        print(f"[ERROR] Lỗi khi hiển thị mẫu: {str(e)}")


def plot_sentiment_distribution(df: pd.DataFrame, save_path: str = "sentiment_distribution.png"):
    """
    Vẽ biểu đồ phân bố sentiment
    
    Args:
        df (pd.DataFrame): DataFrame
        save_path (str): Đường dẫn lưu file
    """
    try:
        print(f"\n[INFO] Vẽ biểu đồ sentiment distribution...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Sentiment Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Sentiment Distribution (Pie Chart)
        ax1 = axes[0, 0]
        sentiment_counts = df['sentiment_label'].value_counts()
        colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
        colors_list = [colors.get(label, '#95a5a6') for label in sentiment_counts.index]
        
        ax1.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            colors=colors_list,
            startangle=90,
            textprops={'fontsize': 10}
        )
        ax1.set_title('Sentiment Distribution', fontweight='bold')
        
        # 2. Sentiment Score Distribution (Histogram)
        ax2 = axes[0, 1]
        ax2.hist(df['sentiment_score'], bins=20, color='#3498db', edgecolor='black', alpha=0.7)
        ax2.axvline(df['sentiment_score'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax2.set_xlabel('Sentiment Score', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('Sentiment Score Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Confidence Distribution by Sentiment
        ax3 = axes[1, 0]
        for label in df['sentiment_label'].unique():
            data = df[df['sentiment_label'] == label]['sentiment_confidence']
            ax3.hist(data, label=label, alpha=0.6, bins=15)
        ax3.set_xlabel('Confidence Score', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Confidence Distribution by Sentiment', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Score vs Confidence Scatter
        ax4 = axes[1, 1]
        scatter_colors = df['sentiment_label'].map(colors)
        ax4.scatter(
            df['sentiment_confidence'],
            df['sentiment_score'],
            c=scatter_colors,
            s=100,
            alpha=0.6,
            edgecolors='black'
        )
        ax4.set_xlabel('Confidence Score', fontweight='bold')
        ax4.set_ylabel('Sentiment Score', fontweight='bold')
        ax4.set_title('Score vs Confidence', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Positive'),
            Patch(facecolor='#e74c3c', label='Negative'),
            Patch(facecolor='#95a5a6', label='Neutral')
        ]
        ax4.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"[SUCCESS] Biểu đồ lưu: {save_path}")
        plt.show()
    
    except Exception as e:
        print(f"[ERROR] Lỗi vẽ biểu đồ: {str(e)}")


def save_results(df: pd.DataFrame, output_file: str = "cafef_news_with_sentiment.csv"):
    """
    Lưu kết quả vào CSV
    
    Args:
        df (pd.DataFrame): DataFrame
        output_file (str): Tên file output
    """
    try:
        print(f"\n[INFO] Lưu kết quả vào {output_file}...")
        
        # Chọn columns
        columns_to_save = [
            'title', 'url', 'publish_date', 'summary',
            'sentiment_label', 'sentiment_confidence', 'sentiment_score'
        ]
        
        # Lọc các columns tồn tại
        columns_to_save = [col for col in columns_to_save if col in df.columns]
        
        df_save = df[columns_to_save].copy()
        df_save.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"[SUCCESS] Lưu {len(df_save)} bài viết vào {output_file}")
        return True
    
    except Exception as e:
        print(f"[ERROR] Lỗi khi lưu: {str(e)}")
        return False


def main():
    """
    Hàm main - điểm vào chính
    """
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS FOR FINANCIAL NEWS USING FINBERT")
    print("="*80)
    
    try:
        # 1. Load news data
        df = load_news_data("csv/cafef_news.csv")
        if df.empty:
            print("[ERROR] Không thể tải dữ liệu tin tức")
            return
        
        # 2. Khởi tạo analyzer
        analyzer = FinBERTSentimentAnalyzer(model_name="ProsusAI/finbert")
        
        # 3. Phân tích sentiment
        df = analyze_news_sentiment(df, analyzer)
        
        # 4. Hiển thị thống kê
        display_sentiment_stats(df)
        
        # 5. Hiển thị mẫu
        display_sample_results(df, num_samples=5)
        
        # 6. Vẽ biểu đồ
        plot_sentiment_distribution(df, save_path="sentiment_distribution.png")
        
        # 7. Lưu kết quả
        save_results(df, output_file="cafef_news_with_sentiment.csv")
        
        print("\n" + "="*80)
        print("[SUCCESS] PHÂN TÍCH HOÀN THÀNH!")
        print("="*80)
        print("Output files:")
        print("  - cafef_news_with_sentiment.csv (Data with sentiment)")
        print("  - sentiment_distribution.png (Visualization)")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
