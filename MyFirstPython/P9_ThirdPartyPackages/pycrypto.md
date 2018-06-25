
https://github.com/dlitz/pycrypto

安裝 ==>$ pip install pycrypto

##### 應用範例:使用AES對稱式密碼加解密
```
from Crypto.Cipher import AES
# Encryption加密
encryption_suite = AES.new('This is a key123', AES.MODE_CBC, 'This is an IV456')
cipher_text = encryption_suite.encrypt("A really secret message. Not for prying eyes.")

# Decryption解密
decryption_suite = AES.new('This is a key123', AES.MODE_CBC, 'This is an IV456')
plain_text = decryption_suite.decrypt(cipher_text)
```

##### 應用範例:使用RSA非對稱式密碼加解密
```
from Crypto import Random
from Crypto.Hash import SHA
from Crypto.Cipher import PKCS1_v1_5 as Cipher_pkcs1_v1_5
from Crypto.Signature import PKCS1_v1_5 as Signature_pkcs1_v1_5
from Crypto.PublicKey import RSA

# 偽亂數產生器
random_generator = Random.new().read
# rsa演算法生成實例
rsa = RSA.generate(1024, random_generator)

# master的秘鑰對的生成
private_pem = rsa.exportKey()

with open('master-private.pem', 'w') as f:
    f.write(private_pem)

public_pem = rsa.publickey().exportKey()
with open('master-public.pem', 'w') as f:
    f.write(public_pem)

# ghost的秘鑰對的生成
private_pem = rsa.exportKey()
with open('master-private.pem', 'w') as f:
    f.write(private_pem)

public_pem = rsa.publickey().exportKey()
with open('master-public.pem', 'w') as f:
    f.write(public_pem)
```
