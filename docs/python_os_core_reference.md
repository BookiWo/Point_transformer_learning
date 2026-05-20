# Python `os` 模块核心函数详解（重构版）

本文件聚焦 `os` / `os.path` 的高频能力，覆盖日常开发中约 80% 的使用场景：路径处理、文件系统操作、权限、环境变量、目录遍历与实用模式。

## 快速导航

1. 路径处理（构建、规范化、拆分）
2. 文件与目录判断（是否存在、类型判断）
3. 目录遍历（`listdir` / `scandir` / `walk`）
4. 创建、删除、重命名
5. 文件属性与权限
6. 工作目录、进程、环境变量
7. 常见实用模式（可直接复用）
8. 易错点与最佳实践

---

## 1. 路径处理函数（最常用）

### 1.1 路径拼接与规范化

| 函数 | 作用 | 示例 | 返回示例 |
| --- | --- | --- | --- |
| `os.path.join()` | 智能拼接路径（跨平台） | `os.path.join('dir', 'sub', 'a.txt')` | `dir/sub/a.txt`（Windows 下为 `dir\\sub\\a.txt`） |
| `os.path.abspath()` | 转绝对路径 | `os.path.abspath('./file')` | `/home/user/proj/file` |
| `os.path.normpath()` | 规范化 `.`、`..`、重复分隔符 | `os.path.normpath('a/../b/./c')` | `b/c` |
| `os.path.relpath()` | 计算相对路径 | `os.path.relpath('/a/b', '/a')` | `b` |

推荐写法：

```python
import os

# 跨平台安全
path = os.path.join("folder", "subfolder", "file.txt")

# 规范化流程（常用于用户输入路径）
clean_path = os.path.normpath(os.path.abspath(os.path.expanduser("~/data/../logs")))
```

不要硬编码分隔符：

```python
# bad
path = "folder" + "/" + "subfolder" + "/" + "file.txt"

# good
path = os.path.join("folder", "subfolder", "file.txt")
```

### 1.2 路径拆分函数

| 函数 | 作用 | 示例 | 返回 |
| --- | --- | --- | --- |
| `os.path.split()` | 拆分目录和文件名 | `os.path.split('/a/b.txt')` | `('/a', 'b.txt')` |
| `os.path.splitext()` | 拆分文件名和扩展名 | `os.path.splitext('a.b.txt')` | `('a.b', '.txt')` |
| `os.path.dirname()` | 取目录部分 | `os.path.dirname('/a/b.txt')` | `/a` |
| `os.path.basename()` | 取文件名部分 | `os.path.basename('/a/b.txt')` | `b.txt` |

---

## 2. 存在性与类型判断

| 函数 | 作用 |
| --- | --- |
| `os.path.exists(path)` | 路径是否存在 |
| `os.path.isfile(path)` | 是否是普通文件 |
| `os.path.isdir(path)` | 是否是目录 |
| `os.path.islink(path)` | 是否是符号链接 |
| `os.path.ismount(path)` | 是否是挂载点 |
| `os.path.isabs(path)` | 是否是绝对路径 |

安全读取模板：

```python
import os

def safe_open_file(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    if not os.path.isfile(filepath):
        raise IsADirectoryError(f"这是目录，不是文件: {filepath}")
    if not os.access(filepath, os.R_OK):
        raise PermissionError(f"没有读取权限: {filepath}")
    return open(filepath, "r", encoding="utf-8")
```

---

## 3. 目录遍历：`listdir` vs `scandir` vs `walk`

| 函数 | 适用场景 | 特点 |
| --- | --- | --- |
| `os.listdir()` | 只想拿名字列表 | 简单，但还要再拼路径并 `stat` |
| `os.scandir()` | 单层高性能遍历 | 推荐，返回 `DirEntry`，少一次系统调用 |
| `os.walk()` | 递归遍历整棵目录树 | 功能强，适合批处理与检索 |

示例：

```python
import os

# 单层遍历（推荐）
with os.scandir(".") as entries:
    for e in entries:
        if e.is_file():
            print("文件", e.name, e.stat().st_size)
        elif e.is_dir():
            print("目录", e.name)

# 递归遍历
for root, dirs, files in os.walk("."):
    print(root, len(dirs), len(files))
```

---

## 4. 创建、删除、重命名

### 4.1 目录操作

| 函数 | 作用 | 注意 |
| --- | --- | --- |
| `os.mkdir(path)` | 创建单层目录 | 父目录必须存在 |
| `os.makedirs(path, exist_ok=True)` | 递归创建目录 | 推荐 |
| `os.rmdir(path)` | 删除空目录 | 非空会报错 |
| `os.removedirs(path)` | 递归删除空目录链 | 仅删除空目录 |

### 4.2 文件与目录重命名/删除

| 函数 | 作用 |
| --- | --- |
| `os.rename(src, dst)` | 重命名或移动 |
| `os.remove(path)` / `os.unlink(path)` | 删除文件 |
| `os.link(src, dst)` | 创建硬链接（类 Unix） |
| `os.symlink(src, dst)` | 创建符号链接 |

删除非空目录请用 `shutil.rmtree()`（危险操作，谨慎）。

---

## 5. 文件属性与权限

### 5.1 文件状态信息

| 函数 | 作用 |
| --- | --- |
| `os.stat(path)` | 获取完整状态 |
| `os.path.getsize(path)` | 文件大小（字节） |
| `os.path.getmtime(path)` | 修改时间戳 |
| `os.path.getatime(path)` | 访问时间戳 |
| `os.path.getctime(path)` | Windows 为创建时间；Unix 常为元数据变更时间 |

```python
import os

st = os.stat("file.txt")
print("size", st.st_size)
print("mode", oct(st.st_mode))
print("mtime", st.st_mtime)
```

### 5.2 权限相关

| 函数 | 作用 | 备注 |
| --- | --- | --- |
| `os.chmod(path, mode)` | 修改权限 | 常用 `0o755`、`0o644` |
| `os.access(path, mode)` | 检查访问权限 | `os.R_OK` / `os.W_OK` / `os.X_OK` |
| `os.chown(path, uid, gid)` | 修改所有者 | 类 Unix |

---

## 6. 工作目录、进程、环境变量

### 6.1 工作目录

| 函数 | 作用 |
| --- | --- |
| `os.getcwd()` | 获取当前工作目录 |
| `os.chdir(path)` | 切换工作目录 |
| `os.path.expanduser('~')` | 展开用户家目录 |

### 6.2 进程与环境变量

| 函数/对象 | 作用 |
| --- | --- |
| `os.getpid()` / `os.getppid()` | 获取当前/父进程 ID |
| `os.getenv(key, default=None)` | 读取环境变量 |
| `os.environ` | 环境变量映射（可读写） |
| `os.system(cmd)` | 执行 shell 命令（简单但能力有限） |

示例：

```python
import os

print("pid", os.getpid())
print("home", os.getenv("HOME"))

os.environ["MY_VAR"] = "demo"
ret = os.system("echo hello")
print("exit code", ret)
```

提示：复杂命令或需要捕获输出时，优先 `subprocess.run()`。

---

## 7. 常用系统信息

| 属性/函数 | 说明 |
| --- | --- |
| `os.name` | 系统类型：`posix`、`nt` 等 |
| `os.sep` | 路径分隔符 |
| `os.pathsep` | PATH 等环境变量分隔符 |
| `os.linesep` | 行分隔符 |
| `os.cpu_count()` | 逻辑 CPU 数 |
| `os.urandom(n)` | 安全随机字节 |

---

## 8. 高级：文件描述符（低级 I/O）

仅在需要底层控制时使用：

| 函数 | 作用 |
| --- | --- |
| `os.open()` / `os.close()` | 打开/关闭文件描述符 |
| `os.read()` / `os.write()` | 基于 fd 读写字节 |
| `os.dup()` | 复制文件描述符 |
| `os.fdopen()` | fd 转文件对象 |

```python
import os

fd = os.open("data.bin", os.O_RDONLY)
try:
    data = os.read(fd, 1024)
finally:
    os.close(fd)
```

---

## 9. 三个可复用实用模式

### 9.1 规范化路径

```python
import os

def normalize_path(path: str) -> str:
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    path = os.path.normpath(path)
    return path
```

### 9.2 确保目录存在

```python
import os

def ensure_directory(path: str, mode: int = 0o755) -> bool:
    if not os.path.exists(path):
        os.makedirs(path, mode=mode, exist_ok=True)
        return True
    if not os.path.isdir(path):
        raise NotADirectoryError(f"路径存在但不是目录: {path}")
    return False
```

### 9.3 递归计算目录大小

```python
import os

def get_directory_size(directory: str) -> int:
    total_size = 0
    for root, _, files in os.walk(directory):
        for name in files:
            p = os.path.join(root, name)
            if os.path.islink(p):
                continue
            try:
                total_size += os.path.getsize(p)
            except OSError:
                continue
    return total_size
```

---

## 10. 易错点与最佳实践

1. 路径拼接始终使用 `os.path.join()`，避免手写 `/` 或 `\\`。
2. 对用户输入路径，建议执行：`expanduser -> abspath -> normpath`。
3. 删除目录前先确认是否非空；非空删除前做白名单检查。
4. `os.path.getctime()` 跨平台语义不同，不要假设它总是“创建时间”。
5. 批量遍历优先 `os.scandir()`（单层）或 `os.walk()`（递归）。
6. 可用 `pathlib.Path` 进一步提升可读性，但理解 `os` 仍是基础。

---

## 11. 一页速查（Cheat Sheet）

| 类别 | 高优先级函数 |
| --- | --- |
| 路径处理 | `join`, `abspath`, `normpath`, `relpath` |
| 路径拆分 | `split`, `splitext`, `dirname`, `basename` |
| 类型判断 | `exists`, `isfile`, `isdir`, `islink` |
| 目录遍历 | `scandir`, `walk` |
| 目录管理 | `makedirs`, `rmdir` |
| 文件管理 | `rename`, `remove` |
| 文件信息 | `stat`, `getsize`, `getmtime` |
| 权限 | `chmod`, `access` |
| 环境变量 | `getenv`, `environ` |
| 工作目录 | `getcwd`, `chdir` |

这份文档适合快速复习与实战查阅。若你后续需要，我可以再补一份“`os` 与 `pathlib` 对照版”用于现代 Python 项目迁移。
