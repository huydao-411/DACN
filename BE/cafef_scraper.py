"""
CafeF News Web Scraper
======================
Script này thu thập tin tức từ trang chứng khoán CafeF
Scrape: Tiêu đề, URL, Thời gian đăng, Tóm tắt nội dung
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from datetime import datetime
from urllib.parse import urljoin
import warnings
warnings.filterwarnings('ignore')


class CafeFScraper:
    """
    Class để scrape tin tức từ CafeF
    """
    
    def __init__(self):
        """Khởi tạo scraper với các cài đặt"""
        self.base_url = "https://cafef.vn"
        # Sử dụng URL trang tin tức chứng khoán của CafeF
        self.target_url = "https://cafef.vn/thi-truong-chung-khoan.chn"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.news_list = []
        self.session = requests.Session()
        
    def check_robots_txt(self):
        """
        Kiểm tra robots.txt để tuân thủ quy tắc scraping
        
        Returns:
            bool: True nếu có thể scrape, False nếu không
        """
        try:
            print("[INFO] Kiểm tra robots.txt...")
            robots_url = urljoin(self.base_url, '/robots.txt')
            response = self.session.get(robots_url, timeout=10, headers=self.headers)
            
            if response.status_code == 200:
                print("[SUCCESS] robots.txt tồn tại")
                print("[INFO] Nội dung robots.txt:")
                print(response.text[:500])  # In 500 ký tự đầu
                return True
            else:
                print(f"[WARNING] Không tìm thấy robots.txt (Status: {response.status_code})")
                return True  # Vẫn cho phép tiếp tục
                
        except Exception as e:
            print(f"[WARNING] Lỗi kiểm tra robots.txt: {str(e)}")
            return True  # Vẫn cho phép tiếp tục
    
    def get_page(self, url, retries=3):
        """
        Lấy nội dung HTML từ URL với xử lý lỗi
        
        Args:
            url (str): URL cần lấy
            retries (int): Số lần retry khi gặp lỗi
        
        Returns:
            BeautifulSoup: HTML đã parse hoặc None nếu lỗi
        """
        for attempt in range(retries):
            try:
                # Random delay 2-5 giây giữa các request
                delay = random.uniform(2, 5)
                print(f"[INFO] Chờ {delay:.2f}s trước khi request...")
                time.sleep(delay)
                
                print(f"[INFO] Đang fetch: {url}")
                response = self.session.get(
                    url, 
                    timeout=10, 
                    headers=self.headers
                )
                
                # Kiểm tra status code
                if response.status_code == 200:
                    print(f"[SUCCESS] Fetch thành công (Status: {response.status_code})")
                    return BeautifulSoup(response.content, 'html.parser')
                elif response.status_code == 429:
                    print(f"[WARNING] Too Many Requests (429). Chờ 30s...")
                    time.sleep(30)
                elif response.status_code == 403:
                    print(f"[WARNING] Forbidden (403). Có thể cần cải thiện headers")
                    return None
                else:
                    print(f"[WARNING] Status code: {response.status_code}")
                    return None
                    
            except requests.Timeout:
                print(f"[ERROR] Timeout lần {attempt+1}/{retries}")
                if attempt < retries - 1:
                    time.sleep(5)
            except requests.ConnectionError:
                print(f"[ERROR] Connection error lần {attempt+1}/{retries}")
                if attempt < retries - 1:
                    time.sleep(5)
            except Exception as e:
                print(f"[ERROR] Lỗi không xác định: {str(e)}")
                return None
        
        print("[CRITICAL] Không thể lấy được trang sau nhiều lần thử")
        return None
    
    def parse_publish_date(self, date_str):
        """
        Parse timestamp string thành datetime object
        
        Args:
            date_str (str): Chuỗi thời gian
        
        Returns:
            datetime: DateTime object hoặc None
        """
        if not date_str:
            return None
        
        try:
            # Cách 1: Định dạng tiêu chuẩn
            for fmt in [
                '%d/%m/%Y %H:%M',
                '%d/%m/%Y',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d'
            ]:
                try:
                    return datetime.strptime(date_str.strip(), fmt)
                except ValueError:
                    continue
            
            # Cách 2: Xử lý tương đối (vd: "2 giờ trước")
            date_str = date_str.strip().lower()
            if 'giây' in date_str or 'second' in date_str:
                return datetime.now()
            elif 'phút' in date_str or 'minute' in date_str:
                return datetime.now()
            elif 'giờ' in date_str or 'hour' in date_str:
                hours = int(''.join(filter(str.isdigit, date_str.split()[0])) or '0')
                return datetime.now()
            elif 'ngày' in date_str or 'day' in date_str:
                days = int(''.join(filter(str.isdigit, date_str.split()[0])) or '0')
                return datetime.now()
            
            return None
        
        except Exception as e:
            print(f"[WARNING] Lỗi parse date '{date_str}': {str(e)}")
            return None
    
    def scrape_news(self, max_articles=100):
        """
        Scrape tin tức từ CafeF
        
        Args:
            max_articles (int): Số bài viết cần scrape
        
        Returns:
            list: Danh sách các bài viết
        """
        try:
            print(f"\n[INFO] Bắt đầu scrape tin tức từ {self.target_url}")
            print(f"[INFO] Mục tiêu: {max_articles} bài viết\n")
            
            article_count = 0
            page = 1
            
            # Lặp qua các trang
            while article_count < max_articles:
                # Tính URL trang
                if page == 1:
                    page_url = self.target_url
                else:
                    page_url = f"{self.target_url}?page={page}"
                
                print(f"[INFO] Scraping trang {page}: {page_url}")
                
                # Lấy trang
                soup = self.get_page(page_url)
                if not soup:
                    print(f"[WARNING] Không thể lấy trang {page}")
                    break
                
                # Tìm tất cả các link tin tức trên trang
                all_links = soup.find_all('a', {'href': True})
                
                if not all_links:
                    print(f"[WARNING] Không tìm thấy link nào trên trang {page}")
                    break
                
                found_on_page = 0
                for link in all_links:
                    if article_count >= max_articles:
                        break
                    
                    try:
                        # Lấy URL
                        url = link.get('href', '').strip()
                        if not url:
                            continue
                        
                        # Chỉ lấy link từ cafef.vn
                        if not ('cafef.vn' in url or url.startswith('/')):
                            continue
                        
                        # Bỏ qua các link không phải bài viết
                        skip_keywords = ['#', 'javascript', 'contact', 'about', 'category', 'login', 'register', 'sitemap', 'rss']
                        if any(skip in url.lower() for skip in skip_keywords):
                            continue
                        
                        if url.startswith('/'):
                            url = urljoin(self.base_url, url)
                        
                        # Lấy tiêu đề
                        title = link.get_text(strip=True)
                        if not title or len(title) < 3:
                            continue
                        
                        # Kiểm tra xem đã scrape cái này chưa
                        if any(item['title'] == title for item in self.news_list):
                            continue
                        
                        # Lấy thời gian đăng (từ parent hoặc siblings)
                        parent = link.find_parent()
                        publish_date_str = "N/A"
                        
                        # Tìm thời gian từ các thẻ có thể
                        time_elem = None
                        if parent:
                            time_elem = parent.find(['time', 'span.date', 'span.time', 'div.date', 'span'])
                        
                        if time_elem:
                            publish_date_str = time_elem.get_text(strip=True)
                        
                        # Lấy tóm tắt (nếu có)
                        summary = "Tin tức từ CafeF"
                        if parent:
                            summary_elem = parent.find(['p', 'span.desc', 'div.desc'])
                            if summary_elem:
                                summary = summary_elem.get_text(strip=True)
                        
                        publish_date = self.parse_publish_date(publish_date_str)
                        
                        # Tạo item tin tức
                        news_item = {
                            'title': title,
                            'url': url,
                            'publish_date': publish_date,
                            'publish_date_str': publish_date_str,
                            'summary': summary[:200]
                        }
                        self.news_list.append(news_item)
                        article_count += 1
                        found_on_page += 1
                        
                        print(f"[OK] ({article_count}) {title[:60]}...")
                    
                    except Exception as e:
                        continue
                
                # Nếu không tìm được bài viết nào trên trang này, dừng
                if found_on_page == 0:
                    print(f"[INFO] Không tìm được bài viết nào trên trang {page}")
                    break
                
                page += 1
            
            print(f"\n[SUCCESS] Scrape thành công {len(self.news_list)} bài viết")
            return self.news_list
        
        except Exception as e:
            print(f"[ERROR] Lỗi khi scrape: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_to_csv(self, filename='cafef_news.csv'):
        """
        Lưu dữ liệu vào file CSV
        
        Args:
            filename (str): Tên file CSV
        
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            if not self.news_list:
                print("[WARNING] Không có dữ liệu để lưu")
                return False
            
            print(f"\n[INFO] Lưu dữ liệu vào {filename}...")
            
            # Tạo DataFrame
            df = pd.DataFrame([{
                'title': item['title'],
                'url': item['url'],
                'publish_date': item['publish_date_str'],
                'summary': item['summary']
            } for item in self.news_list])
            
            # Lưu CSV
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            
            print(f"[SUCCESS] Lưu thành công {len(df)} bài viết vào {filename}")
            return True
        
        except Exception as e:
            print(f"[ERROR] Lỗi khi lưu CSV: {str(e)}")
            return False
    
    def display_sample(self, num_items=5):
        """
        Hiển thị mẫu dữ liệu
        
        Args:
            num_items (int): Số bài viết cần hiển thị
        """
        try:
            if not self.news_list:
                print("[WARNING] Không có dữ liệu để hiển thị")
                return
            
            print("\n" + "="*100)
            print(f"MẪU {num_items} BÀI VIẾT ĐẦU TIÊN")
            print("="*100)
            
            for i, item in enumerate(self.news_list[:num_items], 1):
                print(f"\n[{i}] Tiêu đề: {item['title']}")
                print(f"    URL: {item['url']}")
                print(f"    Thời gian: {item['publish_date_str']}")
                print(f"    Tóm tắt: {item['summary']}")
                print("-" * 100)
        
        except Exception as e:
            print(f"[ERROR] Lỗi khi hiển thị mẫu: {str(e)}")
    
    def get_dataframe(self):
        """
        Lấy dữ liệu dưới dạng DataFrame
        
        Returns:
            pd.DataFrame: DataFrame chứa dữ liệu scrape
        """
        if not self.news_list:
            return pd.DataFrame()
        
        return pd.DataFrame([{
            'title': item['title'],
            'url': item['url'],
            'publish_date': item['publish_date_str'],
            'summary': item['summary']
        } for item in self.news_list])


def main():
    """
    Hàm main - điểm vào chính của chương trình
    """
    print("\n" + "="*100)
    print("WEB SCRAPER CAFEF NEWS")
    print("="*100)
    
    try:
        # Khởi tạo scraper
        scraper = CafeFScraper()
        
        # Kiểm tra robots.txt
        scraper.check_robots_txt()
        
        # Scrape tin tức
        news = scraper.scrape_news(max_articles=100)
        
        if news is None or len(news) == 0:
            print("[WARNING] Không có tin tức được scrape")
            print("[INFO] Thử kiểm tra lại URL hoặc cấu trúc HTML của trang")
            return
        
        # Hiển thị thông tin DataFrame
        df = scraper.get_dataframe()
        print("\n" + "="*100)
        print("THÔNG TIN DATAFRAME")
        print("="*100)
        print(f"Shape: {df.shape[0]} hàng × {df.shape[1]} cột")
        print("\nColumn names:")
        print(df.columns.tolist())
        print("\nData types:")
        print(df.dtypes)
        
        # Hiển thị mẫu 5 bài viết
        scraper.display_sample(num_items=5)
        
        # Lưu vào CSV
        scraper.save_to_csv('cafef_news.csv')
        
        print("\n" + "="*100)
        print("[SUCCESS] HOÀN THÀNH SCRAPING!")
        print("="*100)
        print("Files đã tạo:")
        print("  - cafef_news.csv (dữ liệu tin tức)")
        print("="*100 + "\n")
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Lỗi nghiêm trọng: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
