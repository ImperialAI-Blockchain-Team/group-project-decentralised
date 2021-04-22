import web3 from './web3';
import { jobs} from "../abi/abi";

const address = '0xF127d5453F4871E4D3Ee2E6ebBf559A556aEE306';

export default new web3.eth.Contract(jobs, address);
