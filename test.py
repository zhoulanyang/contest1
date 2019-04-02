from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
# cross_val_score返回交叉验证的每次运行的评分数组, train_test_split用于将数据集划分为训练集和测试集
from sklearn.model_selection import KFold, cross_val_score, train_test_split    # Kfold为K折交叉验证
from sklearn.metrics import mean_squared_error      # 导入计算误差的模块
import xgboost as xgb
import lightgbm as lgb


