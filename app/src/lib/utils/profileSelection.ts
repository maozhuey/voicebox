interface ProfileIdentity {
  id: string;
}

/**
 * Resolve the saved selection against the profiles returned by the currently
 * connected server. A persisted id may belong to a server used in an earlier
 * session, or to a profile that has since been deleted and recreated.
 */
export function resolveSelectedProfileId(
  savedProfileId: string | null,
  profiles: readonly ProfileIdentity[] | undefined,
): string | null {
  if (!profiles) return null;
  if (savedProfileId && profiles.some((profile) => profile.id === savedProfileId)) {
    return savedProfileId;
  }
  return profiles[0]?.id ?? null;
}

/**
 * 编辑弹窗关闭后不能为了保留表单组件而继续读取历史档案。编辑 ID 可能已
 * 被删除或来自另一台服务，此时只会制造无意义的 404；重新打开时才查询。
 */
export function resolveEditingProfileQueryId(
  dialogOpen: boolean,
  editingProfileId: string | null,
): string | null {
  return dialogOpen && editingProfileId ? editingProfileId : null;
}
