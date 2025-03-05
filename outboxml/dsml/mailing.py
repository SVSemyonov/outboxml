import os
from typing import Tuple, List, Dict
import pandas as pd
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from pretty_html_table import build_table


class Mail:
    body_default = """<html><head><meta charset="utf-8"><style> p {font-family: sans-serif, 'Times New Roman', Times, serif;} font {font-family: sans-serif, 'Times New Roman', Times, serif;}</style></head><body>"""

    def __init__(self, config):
        self.config = config
        self.msg = MIMEMultipart()
        self.subject: str = ""
        self.body: str = self.body_default
        self.signature: str = """<p><strong><br />С уважением,<br />Искусственный Интеллект"""
        self.n_photos: int = 0
        self.host = self.config.email_smtp_server
        self.port = self.config.email_port
        self.sender = self.config.email_sender
        self.login = self.config.email_login
        self.password = self.config.email_pass
        self.receivers = self.config.email_receivers
       # self.logo_path = os.path.dirname(__file__) + '/ai_logo.png'

    @staticmethod
    def _image_attachment(path: str, attachment_tag: str):
        w = open(path, 'rb')
        img = MIMEImage(w.read())
        w.close()
        img.add_header('Content-ID', '<{}>'.format(attachment_tag))
        return img

    def add_image(self, bytes_, size_pixel: Tuple[int, int] = (100, 100), n_line_breaks: int = 0):
        """
        Можно добавлять только в формате bytes. Для io.BytesIO: buf.getbuffer().tobytes()

        :param size_pixel: (width, high)
        :param n_line_breaks: number of line breaks after image
        """
        self.body = self.body + '<img src="cid:image{}"high="{}" width="{}"/><br/>'.format(self.n_photos + 1,
                                                                                           size_pixel[1], size_pixel[0])
        img = MIMEImage(bytes_)
        img.add_header('Content-ID', '<{}>'.format('image{}'.format(self.n_photos + 1)))

        self.msg.attach(img)
        self.n_photos = self.n_photos + 1

        self.body = self.body + "<br>" * n_line_breaks

    def add_text(self, text, properties: List = [], n_line_breaks: int = 0):
        """
        :param properties: list of options bold, italic, size:20 (size:+10)
        :param n_line_breaks: number of line breaks after text
        """

        if 'bold' in properties:
            text = "<b>" + text + "</b>"
        if 'italic' in properties:
            text = "<i>" + text + "</i>"
        if any(['size' in i for i in properties]):
            size = ([i for i in properties if 'size' in i][0]).split(':')[1]
            text = '<font size="' + size + '">' + text + '</font>'

        self.body = self.body + '<font>' + text.replace("\n", "<br>") + '</font>' + "<br>" * n_line_breaks

    def add_pandas_table(
            self, table: pd.DataFrame, color: str = "blue_dark", replace_dict: Dict = {}, params: Dict = {}
    ):
        def replace(item, dict_):
            for i in dict_.keys():
                item = item.replace(i, dict_[i])
            return item

        self.body = self.body + replace(build_table(table, color=color, **params), replace_dict)

    def add_line_breaks(self, n_line_breaks: int = 1):
        self.body = self.body + "<br>" * n_line_breaks

    def add_email_subject(self, subject: str):
        self.subject = subject

    def send_mail(self, receiver_emails, add_signature: bool = True):
        if add_signature:
            self.body = self.body + ("%s" % self.signature) + '<img src="cid:ai_logo"high="100" width="100"/><br/>'
       #     self.msg.attach(self._image_attachment(path=self.logo_path, attachment_tag="ai_logo"))

        self.msg["From"] = self.sender
        self.msg["To"] = "; ".join(self.receivers)
        self.msg["Subject"] = self.subject

        self.msg.attach(MIMEText(self.body, "html"))
       # print(self.msg.as_string())
        with smtplib.SMTP(host=self.host, port=self.port) as server:
            server.starttls(context=ssl._create_unverified_context())
            server.login(self.login, self.password)
            server.sendmail(self.sender, self.receivers, self.msg.as_string())
            server.quit()

        self.msg = None
        self.msg = MIMEMultipart()
        self.body = self.body_default
