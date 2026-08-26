import { AlertCircle, Download, RefreshCw } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { useAutoUpdater } from '@/hooks/useAutoUpdater';
import { toChineseErrorMessage } from '@/lib/utils/errorMessage';
import { usePlatform } from '@/platform/PlatformContext';

export function UpdateStatus() {
  const platform = usePlatform();
  const { status, checkForUpdates, downloadAndInstall, restartAndInstall } = useAutoUpdater(false);
  const [currentVersion, setCurrentVersion] = useState<string>('');
  const isDev = !import.meta.env?.PROD;

  useEffect(() => {
    platform.metadata
      .getVersion()
      .then(setCurrentVersion)
      .catch(() => setCurrentVersion('未知'));
  }, [platform]);

  return (
    <Card role="region" aria-label="应用更新" tabIndex={0}>
      <CardHeader>
        <CardTitle>应用更新</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <div className="text-sm font-medium">当前版本</div>
            <div className="text-sm text-muted-foreground">
              v{currentVersion}
              {isDev ? '（开发版）' : ''}
            </div>
          </div>
          {!isDev && (
            <Button
              onClick={checkForUpdates}
              disabled={status.checking || status.downloading || status.readyToInstall}
              variant="outline"
              size="sm"
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${status.checking ? 'animate-spin' : ''}`} />
              检查更新
            </Button>
          )}
        </div>

        {isDev ? (
          <div className="text-sm text-muted-foreground">开发模式下已停用自动更新。</div>
        ) : (
          <>
            {status.checking && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <RefreshCw className="h-4 w-4 animate-spin" />
                正在检查更新…
              </div>
            )}

            {status.error && (
              <div className="flex items-center gap-2 text-sm text-destructive">
                <AlertCircle className="h-4 w-4" />
                {toChineseErrorMessage(status.error, '检查更新失败')}
              </div>
            )}

            {status.available && !status.downloading && !status.readyToInstall && (
              <div className="space-y-3 p-4 border rounded-lg bg-primary/5">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-semibold">发现新版本</div>
                    <div className="text-sm text-muted-foreground">版本 {status.version}</div>
                  </div>
                  <Badge>新版本</Badge>
                </div>
                <Button onClick={downloadAndInstall} className="w-full" size="sm">
                  <Download className="h-4 w-4 mr-2" />
                  下载更新
                </Button>
              </div>
            )}

            {status.downloading && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <Download className="h-4 w-4" />
                    正在下载更新…
                  </div>
                  {status.downloadProgress !== undefined && (
                    <span className="text-muted-foreground">{status.downloadProgress}%</span>
                  )}
                </div>
                <Progress value={status.downloadProgress} />
                {status.downloadedBytes !== undefined &&
                  status.totalBytes !== undefined &&
                  status.totalBytes > 0 && (
                    <div className="text-xs text-muted-foreground">
                      {(status.downloadedBytes / 1024 / 1024).toFixed(1)} MB /{' '}
                      {(status.totalBytes / 1024 / 1024).toFixed(1)} MB
                    </div>
                  )}
              </div>
            )}

            {status.readyToInstall && (
              <div className="space-y-3 p-4 border rounded-lg bg-accent/30 border-accent/50">
                <div className="flex items-center gap-2">
                  <div>
                    <div className="font-semibold">更新已可安装</div>
                    <div className="text-sm text-muted-foreground">
                      版本 {status.version} 已下载
                    </div>
                  </div>
                </div>
                <div className="text-sm text-muted-foreground">
                  应用需要重启才能完成安装，您可以现在重启，也可以稍后处理。
                </div>
                <Button onClick={restartAndInstall} className="w-full" size="sm">
                  <RefreshCw className="h-4 w-4 mr-2" />
                  立即重启
                </Button>
              </div>
            )}

            {!status.available && !status.checking && !status.error && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                当前已是最新版本
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
