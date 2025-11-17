# auth.py - 用户认证和授权模块
import os
import hashlib
import secrets
from functools import wraps
from flask import Flask, session, request, jsonify, redirect, url_for
from typing import Optional, Dict, Any

# 尝试导入Flask-Login，如果失败则使用备用实现
try:
    from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
    HAS_FLASK_LOGIN = True
except ImportError:
    HAS_FLASK_LOGIN = False
    print("警告: Flask-Login库未安装，使用备用认证实现")
    
    # 备用实现
    class UserMixin:
        pass
    
    class LoginManager:
        def __init__(self):
            self.user_callback = None
            self.login_view = None
            self.login_message = None
        
        def init_app(self, app):
            pass
        
        def user_loader(self, callback):
            self.user_callback = callback
            return callback
    
    # 全局变量存储当前用户
    _current_user = None
    
    def login_user(user):
        global _current_user
        _current_user = user
        session['user_id'] = user.id
        return True
    
    def logout_user():
        global _current_user
        _current_user = None
        session.pop('user_id', None)
    
    def login_required(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return jsonify({'error': '未登录'}), 401
            return f(*args, **kwargs)
        return decorated_function
    
    class current_user:
        @property
        def is_authenticated(self):
            return 'user_id' in session
        
        @property
        def id(self):
            return session.get('user_id')
        
        @property
        def username(self):
            return session.get('username')
        
        @property
        def role(self):
            return session.get('role')

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USERS_FILE = os.path.join(BASE_DIR, 'users.json')

class User(UserMixin):
    """用户类"""
    def __init__(self, user_id: str, username: str, email: str, role: str = 'user'):
        self.id = user_id
        self.username = username
        self.email = email
        self.role = role
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role
        }

class AuthManager:
    """认证管理器"""
    def __init__(self, app: Optional[Flask] = None):
        self.app = app
        self.login_manager = LoginManager()
        self.users: Dict[str, User] = {}
        self._init_default_admin()
        
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """初始化Flask应用"""
        self.app = app
        self.login_manager.init_app(app)
        self.login_manager.login_view = 'login'
        self.login_manager.login_message = '请先登录'
        self.login_manager.login_message_category = 'info'
        
        @self.login_manager.user_loader
        def load_user(user_id: str) -> Optional[User]:
            return self.users.get(user_id)
        
        # 加载用户数据
        self._load_users()
    
    def _init_default_admin(self):
        """初始化默认管理员账户"""
        admin_id = 'admin_001'
        admin_user = User(
            user_id=admin_id,
            username='admin',
            email='admin@example.com',
            role='admin'
        )
        self.users[admin_id] = admin_user
    
    def _load_users(self):
        """从文件加载用户数据"""
        try:
            import json
            if os.path.exists(USERS_FILE):
                with open(USERS_FILE, 'r', encoding='utf-8') as f:
                    users_data = json.load(f)
                    for user_data in users_data:
                        user = User(**user_data)
                        self.users[user.id] = user
        except Exception as e:
            print(f"加载用户数据失败: {e}")
    
    def _save_users(self):
        """保存用户数据到文件"""
        try:
            import json
            users_data = [user.to_dict() for user in self.users.values()]
            with open(USERS_FILE, 'w', encoding='utf-8') as f:
                json.dump(users_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存用户数据失败: {e}")
    
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """用户认证"""
        # 默认管理员账户
        if username == 'admin' and password == 'admin123':
            user = self.users.get('admin_001')
            if user:
                login_user(user)
                if not HAS_FLASK_LOGIN:
                    # 备用实现需要手动设置session
                    session['username'] = user.username
                    session['role'] = user.role
                return user
        
        # 其他用户认证逻辑
        for user in self.users.values():
            if user.username == username:
                # 这里应该验证密码哈希，简化示例直接返回
                login_user(user)
                if not HAS_FLASK_LOGIN:
                    # 备用实现需要手动设置session
                    session['username'] = user.username
                    session['role'] = user.role
                return user
        
        return None
    
    def create_user(self, username: str, email: str, password: str, role: str = 'user') -> Optional[User]:
        """创建新用户"""
        user_id = f"user_{secrets.token_hex(8)}"
        
        # 检查用户名是否已存在
        for user in self.users.values():
            if user.username == username:
                return None
        
        # 创建用户
        new_user = User(user_id, username, email, role)
        self.users[user_id] = new_user
        self._save_users()
        
        return new_user
    
    def logout_user(self):
        """用户登出"""
        logout_user()
    
    def get_current_user(self) -> Optional[User]:
        """获取当前用户"""
        if HAS_FLASK_LOGIN:
            return current_user if current_user.is_authenticated else None
        else:
            # 备用实现
            if 'user_id' in session:
                user_id = session['user_id']
                return self.users.get(user_id)
            return None

# 权限装饰器
def admin_required(f):
    """管理员权限装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if HAS_FLASK_LOGIN:
            if not current_user.is_authenticated:
                return jsonify({'error': '未登录'}), 401
            if current_user.role != 'admin':
                return jsonify({'error': '权限不足'}), 403
        else:
            # 备用实现
            if 'user_id' not in session:
                return jsonify({'error': '未登录'}), 401
            if session.get('role') != 'admin':
                return jsonify({'error': '权限不足'}), 403
        
        return f(*args, **kwargs)
    return decorated_function

def user_required(f):
    """用户权限装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if HAS_FLASK_LOGIN:
            if not current_user.is_authenticated:
                return jsonify({'error': '未登录'}), 401
        else:
            # 备用实现
            if 'user_id' not in session:
                return jsonify({'error': '未登录'}), 401
        return f(*args, **kwargs)
    return decorated_function

# 全局认证管理器实例
auth_manager = AuthManager()