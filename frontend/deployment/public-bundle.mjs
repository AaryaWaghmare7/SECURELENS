// React Router uses this dummy URL only for parsing without a browser location.
// Keep a non-routable public-form base in production, not a local service URL.
export function publicRouterBase() {
  return {
    name: 'securelens-public-router-base',
    apply: 'build',
    transform(code, id) {
      if (!id.replaceAll('\\', '/').includes('/react-router/dist/')) return null;
      return {
        code: code.replace(/(["'`])http:\/\/localhost\1/g, '$1https://router.invalid$1'),
        map: null,
      };
    },
  };
}
