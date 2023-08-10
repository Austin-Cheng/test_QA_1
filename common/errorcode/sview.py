import copy
import re

from common.errorcode import errDict


class Sview(object):
    """
    api对外暴露统一的接口
    """

    @staticmethod
    def replaceArgs(content, ds):
        p = re.compile('\[(\w+)\]')
        args = p.findall(content)
        if len(args) <= 0:
            return content
        for arg in args:
            subStr = "[{}]".format(arg)
            content = str(content).replace(subStr, str(ds[arg]))
        return content

    @classmethod
    def TErrorreturn(cls, ErrorCode, **args):
        detailError = errDict[ErrorCode]
        detailErrorCopy = copy.deepcopy(detailError)

        detailErrorCopy['description'] = cls.replaceArgs(detailError['description'], args)
        detailErrorCopy['cause'] = cls.replaceArgs(detailError['cause'], args)
        detailErrorCopy['solution'] = cls.replaceArgs(detailError['solution'], args)
        return detailErrorCopy

    @staticmethod
    def json_return(data):
        """
        统一返回调用信息
        :param data:
        :return:
        """
        obj = dict()
        obj['result'] = data
        return obj
