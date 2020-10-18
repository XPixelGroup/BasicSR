import requests


def baidu_decode_url(encrypted_url):
    """Decrypt baidu ecrypted url."""
    url = encrypted_url
    map1 = {'_z2C$q': ':', '_z&e3B': '.', 'AzdH3F': '/'}
    map2 = {
        'w': 'a', 'k': 'b', 'v': 'c', '1': 'd', 'j': 'e',
        'u': 'f', '2': 'g', 'i': 'h', 't': 'i', '3': 'j',
        'h': 'k', 's': 'l', '4': 'm', 'g': 'n', '5': 'o',
        'r': 'p', 'q': 'q', '6': 'r', 'f': 's', 'p': 't',
        '7': 'u', 'e': 'v', 'o': 'w', '8': '1', 'd': '2',
        'n': '3', '9': '4', 'c': '5', 'm': '6', '0': '7',
        'b': '8', 'l': '9', 'a': '0'
    }  # yapf: disable
    for (ciphertext, plaintext) in map1.items():
        url = url.replace(ciphertext, plaintext)
    char_list = [char for char in url]
    for i in range(len(char_list)):
        if char_list[i] in map2:
            char_list[i] = map2[char_list[i]]
    url = ''.join(char_list)
    return url


def setup_session():
    headers = {
        'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_3)'
                       ' AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/48.0.2564.116 Safari/537.36')
    }
    session = requests.Session()
    session.headers.update(headers)
    return session
